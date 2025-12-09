from pathlib import Path
import json
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
import torch


class MoETrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.teacher_model = None
        
        super().__init__(cfg, overrides, _callbacks)
        
        self.disable_val_loss = True
        self.moe_domain_split = getattr(self.args, "moe_domain_split", 35)
        self.usage_history = []

        # 콜백 등록 (BaseTrainer.run_callbacks(event) -> callback(self))
        cb = self.callbacks
        cb.setdefault("on_train_start", []).append(lambda t: t._moe_on_train_start())
        cb.setdefault("on_train_epoch_start", []).append(lambda t: t._moe_on_train_epoch_start())
        cb.setdefault("on_train_batch_end", []).append(lambda t: t._moe_on_train_batch_end())
        cb.setdefault("on_train_epoch_end", []).append(lambda t: t._moe_on_train_epoch_end())
        cb.setdefault("on_train_end", []).append(lambda t: t._moe_on_train_end())


    def get_model(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import DetectionModel
        if weights is not None:
            print("[MoETrainer] get_model: reuse provided weights")
            model = weights
        else:
            print("[MoETrainer] get_model: build new model from cfg (fallback)")
            model = DetectionModel(cfg, verbose=verbose)

        
        # if self.teacher_model is None:
        #     teacher_ckpt = "/ultralytics/data/teacher_v1/best.pt"
        #     y = YOLO(teacher_ckpt)
        #     t_model = y.model
        #     for p in t_model.parameters():
        #         p.requires_grad = False
        #     t_model.eval()
        #     self.teacher_model = t_model
        #     print("[MoETrainer] teacher loaded:", teacher_ckpt)

        #     model.teacher_model = self.teacher_model
        #     print("[MoETrainer] teacher attached to model")

        return model

    def _get_head_usage_tensors(self):
        """
        head.expert_counts / expert_counts_nonmoving / expert_counts_rider
        를 모두 CPU float list로 안전하게 변환해서 리턴.
        (없으면 None)
        """
        head = self.model.model[-1]

        def _to_list(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            # 혹시 텐서가 아니더라도 리스트로 캐스팅 가능한 경우 방어적 처리
            try:
                return list(x)
            except TypeError:
                return None

        g = getattr(head, "expert_counts", None)
        n = getattr(head, "expert_counts_nonmoving", None)
        r = getattr(head, "expert_counts_rider", None)

        return _to_list(g), _to_list(n), _to_list(r)

    def _set_router_temperature(self):
        epoch = getattr(self, "epoch", 0)
        max_epoch = self.epochs

        if epoch < 10:
            T = 1.5
        elif epoch < 30:
            T = 1.2
        else:
            T = 1.0

        head = self.model.model[-1]
        if not hasattr(head, "routers"):
            return

        for router in head.routers:
            if hasattr(router, "set_temperature"):
                router.set_temperature(T)

    def _moe_on_train_start(self):
        head = self.model.model[-1]
        if hasattr(head, "reset_usage"):
            head.reset_usage()
        self.usage_history.clear()

    def _moe_on_train_epoch_start(self):
        head = self.model.model[-1]
        if hasattr(head, "reset_usage"):
            head.reset_usage()

        self._set_router_temperature()

    def _moe_on_train_batch_end(self):
        head = self.model.model[-1]
        if not hasattr(head, "_last_router_info"):
            return
        routing_info = head._last_router_info
        if routing_info is None:
            return

        batch = getattr(self, "batch", None)
        if batch is None:
            return

        self._update_domain_usage(batch, head, routing_info)

    def _moe_on_train_epoch_end(self):
        self._log_usage_epoch(self.epoch)

    def _moe_on_train_end(self):
        # 마지막 epoch도 한 번 더 찍어주고 싶다면 여기에서 print 가능
        if self.usage_history:
            last = self.usage_history[-1]
            e = last["epoch"] + 1

            g_print = torch.tensor(last["global"])
            print(f"[MoETrainer] Final epoch {e} usage:")
            print("  Global   :", g_print)

            if last["nonmoving"] is not None:
                n_print = torch.tensor(last["nonmoving"])
                print("  Nonmoving:", n_print)

            if last["rider"] is not None:
                r_print = torch.tensor(last["rider"])
                print("  Rider    :", r_print)

        save_dir = getattr(self, "save_dir", None)
        if save_dir is None:
            return

        # out = []
        # for rec in self.usage_history:
        #     item = {
        #         "epoch": int(rec["epoch"]),
        #         "global": rec["global"].tolist(),
        #         "nonmoving": rec["nonmoving"].tolist() if rec["nonmoving"] is not None else None,
        #         "rider": rec["rider"].tolist() if rec["rider"] is not None else None,
        #     }
        #     out.append(item)

        path = Path(save_dir) / "moe_usage.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.usage_history, f, ensure_ascii=False, indent=2)

        print(f"[MoETrainer] Saved MoE usage history to {path}")

    def _update_domain_usage(self, batch, head, routing_info):
        # 4-1) global usage는 head.forward에서 이미 expert_counts에 누적됨
        #      여기서는 nonmoving / rider만 처리
        if self.moe_domain_split is None:
            return

        if "cls" not in batch or "batch_idx" not in batch:
            return

        cls = batch["cls"].squeeze(-1)         # [M]
        bidx = batch["batch_idx"].long()       # [M]
        B = int(bidx.max().item()) + 1

        with torch.no_grad():
            for b in range(B):
                mask = (bidx == b)
                if not mask.any():
                    continue

                cls_b = cls[mask]
                has_non = (cls_b < self.moe_domain_split).any()
                has_rid = (cls_b >= self.moe_domain_split).any()

                if not (has_non or has_rid):
                    continue

                for routing_weights, _ in routing_info:  # [B,E]
                    if b >= routing_weights.shape[0]:
                        continue
                    usage = routing_weights[b]  # [E]

                    if has_non and hasattr(head, "expert_counts_nonmoving"):
                        head.expert_counts_nonmoving += usage
                    if has_rid and hasattr(head, "expert_counts_rider"):
                        head.expert_counts_rider += usage

    # --------------------------------------------------
    # 5) epoch별 usage 로깅 / 저장
    # --------------------------------------------------
    def _log_usage_epoch(self, epoch_idx: int):
        head = self.model.model[-1]
        if not hasattr(head, "expert_counts"):
            return

        g = head.expert_counts
        n = getattr(head, "expert_counts_nonmoving", None)
        r = getattr(head, "expert_counts_rider", None)

        record = {
            "epoch": epoch_idx,
            "global": g.detach().cpu().tolist(),
            "nonmoving": n.detach().cpu().tolist() if n is not None else None,
            "rider": r.detach().cpu().tolist() if r is not None else None,
        }
        self.usage_history.append(record)

        epoch_num = epoch_idx + 1
        if epoch_num % 1 == 0:  # 지금은 매 epoch 출력, 10단위만 보고 싶으면 %10==0 으로
            g_print = torch.tensor(record["global"])
            print(f"\n[Usage] Epoch {epoch_num}")
            print("  Global   :", g_print)

            if record["nonmoving"] is not None:
                n_print = torch.tensor(record["nonmoving"])
                print("  Nonmoving:", n_print)
            if record["rider"] is not None:
                r_print = torch.tensor(record["rider"])
                print("  Rider    :", r_print)
