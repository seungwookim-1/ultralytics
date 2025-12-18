from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
import torch
from .moe_usage_logger import MoEUsageLogger


class MoETrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.teacher_model = None
        super().__init__(cfg, overrides, _callbacks)

        self._last_batch = None
        self.disable_val_loss = True

        # 추후 동적으로 split 구현
        self.moe_domain_split = getattr(self.args, "moe_domain_split", 35)

        self.usage_logger = MoEUsageLogger(
            trainer=self,
            moe_domain_split=self.moe_domain_split,
        )

        # 콜백 등록 (BaseTrainer.run_callbacks(event) -> callback(self))
        cb = self.callbacks
        cb.setdefault("on_train_start", []).append(
            lambda t: t._moe_on_train_start()
        )
        cb.setdefault("on_train_epoch_start", []).append(
            lambda t: t._moe_on_train_epoch_start()
        )
        cb.setdefault("on_train_batch_end", []).append(
            lambda t: t._moe_on_train_batch_end()
        )
        cb.setdefault("on_train_epoch_end", []).append(
            lambda t: t._moe_on_train_epoch_end(t.epoch)
        )
        cb.setdefault("on_train_end", []).append(
            lambda t: t._moe_on_train_end()
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(cfg, torch.nn.Module):
            print("[MoETrainer] get_model: cfg is nn.Module -> reuse cfg")
            return cfg
        if isinstance(weights, torch.nn.Module):
            print("[MoETrainer] get_model: reuse provided weights")
            return weights

        print("[MoETrainer] get_model: build new model from cfg (fallback)")
        return DetectionModel(cfg, verbose=verbose)
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

        # return model

    def _moe_on_train_start(self):
        self._apply_inital_moe_params()
        self.usage_logger.on_train_start()

    def _moe_on_train_epoch_start(self):
        self._apply_moe_params_schedule()
        self.usage_logger.on_train_epoch_start()      

    def _moe_on_train_batch_end(self):
        head = self.model.model[-1]
        if not hasattr(head, "_last_router_info"):
            return
        routing_info = head._last_router_info
        if routing_info is None:
            return

        batch = getattr(self, "_last_batch", None)
        if batch is None:
            return

        self.usage_logger.on_train_batch_end(batch, routing_info)

    def _moe_on_train_epoch_end(self, epoch_idx: int):
        self.usage_logger.on_train_epoch_end(epoch_idx)       

    def _moe_on_train_end(self):
        self.usage_logger.on_train_end()       


    def _apply_inital_moe_params(self):
        # print(self.model)
        head = self.model.model[-1]
        # params = getattr(self.__class__, "MOE_PARAMS", None)
        params = getattr(self.model, "moe_params", None)
        assert params is not None, "MoETrainer.MOE_PARAMS must be set before train()"

        head.lambda_entropy = float(params.lambda_entropy)
        head.lambda_balance = float(params.lambda_balance)
        head.noise_scale = float(params.noise_scale)
        head.aux_loss_weight = float(params.aux_loss_weight)

        print("[MoETrainer] applied:",
              "lambda_entropy", head.lambda_entropy,
              "lambda_balance", head.lambda_balance,
              "noise_scale", head.noise_scale,
              "aux_loss_weight", head.aux_loss_weight,
        )

    def _apply_moe_params_schedule(self):
        from math import cos, pi, exp

        # def decay(p, k=2.0):
        #     return max(0.0, (1.0 - p) ** k)

        head = self.model.model[-1]
        # params = getattr(self.__class__, "MOE_PARAMS", None)
        params = getattr(self.model, "moe_params", None)
        sched  = getattr(self.model, "moe_schedule", {})

        epoch = getattr(self, "epoch", 0)
        max_epoch = getattr(self, "epochs", None)
        assert max_epoch and max_epoch > 0
        assert params is not None


        # progress: 0..1
        p = epoch / max(1, (max_epoch - 1))
        s = 0.5 * (1 + cos(pi * p))  # 1 -> 0

        def peak(p, mu=0.48, sigma=0.10):
            return exp(-((p - mu) ** 2) / (2 * (sigma ** 2)))

        # ---- base values ----
        base_entropy = float(params.lambda_entropy)
        base_balance = float(params.lambda_balance)
        base_noise   = float(params.noise_scale)
        base_aux     = float(params.aux_loss_weight)

        def mult(key: str, default_min: float, default_max: float) -> float:
            cfg = sched.get(key, None)
            if cfg is None:
                m_min, m_max = default_min, default_max
            else:
                m_min = float(cfg.get("m_min", default_min))
                m_max = float(cfg.get("m_max", default_max))
            return m_min + (m_max - m_min) * s

        def mult_peak(key: str, default_min: float, default_max: float,
                    mu: float = 0.48, sigma: float = 0.10, peak_gain: float = 0.0) -> float:
            m = mult(key, default_min, default_max)        # cos 기반(전체 1->0)
            pk = peak(p, mu=mu, sigma=sigma)               # 중반만 0~1
            return m * (1.0 + peak_gain * pk)

        # ---- per-param multipliers (단일 파라미터만 실험하려면 나머지 m_min=m_max=1로 두면 됨) ----
        m_ent = mult("lambda_entropy", default_min=1.0, default_max=1.0)
        m_bal = mult("lambda_balance", default_min=0.30, default_max=1.3)
        m_ns  = mult_peak("noise_scale",
                default_min=0.08, default_max=2.0,
                mu=0.48, sigma=0.10, peak_gain=1.0)
        # aux는 우선 고정 권장. 스케줄 실험하고 싶으면 아래처럼:
        m_aux = mult("aux_loss_weight", default_min=1.0, default_max=1.0)

        head.lambda_entropy = base_entropy * m_ent
        head.lambda_balance = base_balance * m_bal
        head.noise_scale    = base_noise   * m_ns
        head.aux_loss_weight = base_aux * m_aux

        # ---- temperature schedule ----
        # base T=1.0 기준. 필요하면 sched["temperature"]로 제어
        t_cfg = sched.get("temperature", {})
        T_min = float(t_cfg.get("min", 1.0))
        T_max = float(t_cfg.get("max", 1.5))
        T = T_min + (T_max - T_min) * s

        T_peak_gain = float(t_cfg.get("peak_gain", 0.4))   # 0.3~0.6 권장
        T *= (1.0 + T_peak_gain * peak(p, mu=0.48, sigma=0.10))

        if not hasattr(head, "routers"):
            return

        for router in head.routers:
            if hasattr(router, "set_temperature"):
                router.set_temperature(T)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._last_batch = batch
        return batch