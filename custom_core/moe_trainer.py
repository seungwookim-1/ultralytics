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
            lambda t: t.usage_logger.on_train_start()
        )
        cb.setdefault("on_train_epoch_start", []).append(
            lambda t: t.usage_logger.on_train_epoch_start()
        )
        cb.setdefault("on_train_batch_end", []).append(
            lambda t: t._moe_on_train_batch_end()
        )
        cb.setdefault("on_train_epoch_end", []).append(
            lambda t: t.usage_logger.on_train_epoch_end(t.epoch)
        )
        cb.setdefault("on_train_end", []).append(
            lambda t: t.usage_logger.on_train_end()
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
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

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._last_batch = batch
        return batch

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

        # 실제 도메인 usage 업데이트는 logger에 위임
        self.usage_logger.on_train_batch_end(batch, routing_info)
