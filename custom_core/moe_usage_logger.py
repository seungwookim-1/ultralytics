import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, List
import numpy as np
import torch
from .dataclass.usage_logger import MoEUsageRecord


@dataclass
class MoEUsageLogger:
    def __init__(self, trainer: Any, moe_domain_split: int | None = None):
        self.trainer = trainer
        self.moe_domain_split = moe_domain_split
        self.usage_history: List[MoEUsageRecord] = []

    # --------------------------------------------------
    # 0) 내부 유틸
    # --------------------------------------------------
    @property
    def head(self):
        return self.trainer.model.model[-1]

    def _to_list(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        try:
            return list(x)
        except TypeError:
            return None

    # Callback Entry Points
    def on_train_start(self):
        if getattr(self, "_started", False):
            return
        self._started = True
        if hasattr(self.head, "reset_usage"):
            self.head.reset_usage()
        self.usage_history.clear()
        print("[MoETrainer] MoE usage history cleared")

    def on_train_epoch_start(self):
        epoch = getattr(self.trainer, "epoch", None)
        if epoch is not None and getattr(self, "_last_reset_epoch", None) == epoch:
            return
        self._last_reset_epoch = epoch
        if hasattr(self.head, "reset_usage"):
            self.head.reset_usage()

    def on_train_batch_end(self, batch, routing_info):
        if routing_info is None:
            return
        self._update_domain_usage(batch, routing_info)

    def on_train_epoch_end(self, epoch_idx: int):
        self._log_usage_epoch(epoch_idx)

    def on_train_end(self):
        self._save_usage_history()


    def _update_domain_usage(self, batch, routing_info):
        if self.moe_domain_split is None:
            print("[MoETrainer] _update_domain_usage : domain_split is None")
            return

        if not isinstance(batch, dict):
            print("[MoETrainer] _update_domain_usage : batch is not dict")
            return

        if "cls" not in batch or "batch_idx" not in batch:
            print("[MoETrainer] _update_domain_usage : cls or batch_idx not in batch")
            return

        cls = batch["cls"]
        bidx = batch["batch_idx"]

        # --- numpy → torch 방어 ---
        import numpy as np
        if isinstance(cls, np.ndarray):
            cls = torch.as_tensor(cls)
        if isinstance(bidx, np.ndarray):
            bidx = torch.as_tensor(bidx)

        if not isinstance(cls, torch.Tensor) or not isinstance(bidx, torch.Tensor):
            print(f"[MoETrainer] _update_domain_usage : unexpected types "
                    f"cls={type(cls)}, bidx={type(bidx)} → skip")
            return

        # --- shape 정규화: 항상 [M] ---
        if cls.ndim == 2 and cls.shape[1] == 1:
            cls = cls[:, 0]
        elif cls.ndim == 0:
            cls = cls.view(1)
        elif cls.ndim != 1:
            print(f"[MoETrainer] WARNING: unexpected cls shape {cls.shape}, "
                    "reshape→take first column")
            cls = cls.reshape(cls.shape[0], -1)[:, 0]

        if bidx.ndim == 2 and bidx.shape[1] == 1:
            bidx = bidx[:, 0]
        elif bidx.ndim == 0:
            bidx = bidx.view(1)
        elif bidx.ndim != 1:
            print(f"[MoETrainer] WARNING: unexpected batch_idx shape {bidx.shape}, "
                    "reshape→take first column")
            bidx = bidx.reshape(bidx.shape[0], -1)[:, 0]

        if cls.shape[0] != bidx.shape[0]:
            print(f"[MoETrainer] WARNING: len(cls)={cls.shape[0]} != "
                    f"len(bidx)={bidx.shape[0]} → skip")
            return

        cls = cls.long()
        bidx = bidx.long()

        if cls.numel() == 0:
            print("[MoETrainer] _update_domain_usage : empty cls → skip")
            return

        # === ★ 여기에서 B를 weights 기준으로 맞춘다 ★ ===
        # routing_info: list of (weights, router_out) per scale
        B_router = None
        for weights, _ in routing_info:
            if isinstance(weights, torch.Tensor):
                B_router = weights.shape[0]
                break

        if B_router is None:
            print("[MoETrainer] _update_domain_usage : no valid weights in routing_info")
            return

        # 참고: bidx.max()+1 은 "GT가 있는 이미지 수"일 뿐이라
        # GT 없는 이미지를 포함한 실제 배치 크기보다 작을 수 있다.
        # B_gt = int(bidx.max().item()) + 1
        # print(f"[MoETrainer] B_router={B_router}, B_gt={B_gt}")

        # --- device 정렬 ---
        if hasattr(self.head, "expert_counts"):
            ref_device = self.head.expert_counts.device
        else:
            ref_device = cls.device

        cls = cls.to(ref_device)
        bidx = bidx.to(ref_device)

        # === ★ B_router 기준으로 플래그 생성 ★ ===
        has_non = torch.zeros(B_router, dtype=torch.bool, device=ref_device)
        has_rid = torch.zeros(B_router, dtype=torch.bool, device=ref_device)

        # 각 이미지 b(0 ~ B_router-1)에 대해 GT가 있으면 도메인 플래그 설정
        for b in range(B_router):
            mask = (bidx == b)
            if not mask.any():
                continue  # 이 이미지는 GT가 없는 이미지

            cls_b = cls[mask]
            has_non[b] = (cls_b < self.moe_domain_split).any()
            has_rid[b] = (cls_b >= self.moe_domain_split).any()

        # --- 실제 usage 누적 ---
        with torch.no_grad():
            for weights, _ in routing_info:  # weights: [B_router, E]
                if not isinstance(weights, torch.Tensor):
                    continue

                if weights.device != ref_device:
                    weights = weights.to(ref_device)

                # nonmoving
                if has_non.any() and hasattr(self.head, "expert_counts_nonmoving"):
                    self.head.expert_counts_nonmoving += weights[has_non].sum(dim=0)

                # rider
                if has_rid.any() and hasattr(self.head, "expert_counts_rider"):
                    self.head.expert_counts_rider += weights[has_rid].sum(dim=0)

    # --------------------------------------------------
    # 5) epoch별 usage 로깅 / 저장
    # --------------------------------------------------
    def _log_usage_epoch(self, epoch_idx: int):
        if not hasattr(self.head, "expert_counts"):
            return

        g = self._to_list(getattr(self.head, "expert_counts", None))
        n = self._to_list(getattr(self.head, "expert_counts_nonmoving", None))
        r = self._to_list(getattr(self.head, "expert_counts_rider", None))

        if g is None:
            return

        record = MoEUsageRecord(
            epoch=epoch_idx,
            global_usage=g,
            nonmoving_usage=n,
            rider_usage=r,
        )
        self.usage_history.append(record)

        epoch_num = epoch_idx + 1
        print(f"\n[Usage] Epoch {epoch_num}")
        print("  Global   :", torch.tensor(record.global_usage))
        if record.nonmoving_usage is not None:
            print("  Nonmoving:", torch.tensor(record.nonmoving_usage))
        if record.rider_usage is not None:
            print("  Rider    :", torch.tensor(record.rider_usage))

    def _save_usage_history(self):
        if self.usage_history:
            last = self.usage_history[-1]
            last_epoch = last.epoch + 1

            g_print = torch.tensor(last.global_usage)
            print(f"[MoETrainer] Final epoch {last_epoch} usage:")
            print("  Global   :", g_print)

            if last.nonmoving_usage is not None:
                n_print = torch.tensor(last.nonmoving_usage)
                print("  Nonmoving:", n_print)

            if last.rider_usage is not None:
                r_print = torch.tensor(last.rider_usage)
                print("  Rider    :", r_print)

        save_dir = getattr(self.trainer, "save_dir", None)
        if save_dir is None:
            print(f"[MoETrainer] Failed to save MoE usage history : save_dir is None")    
            return

        path = Path(save_dir) / "moe_usage.json"
        usage_history_dict = [asdict(record) for record in self.usage_history]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(usage_history_dict, f, ensure_ascii=False, indent=2)

        print(f"[MoETrainer] Saved MoE usage history to {path}")