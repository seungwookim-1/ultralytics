from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG


class MoETrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.teacher_model = None
        super().__init__(cfg, overrides, _callbacks)


    def get_model(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import DetectionModel
        if weights is not None:
            print("[MoETrainer] get_model: reuse provided weights")
            model = weights
        else:
            print("[MoETrainer] get_model: build new model from cfg (fallback)")
            model = DetectionModel(cfg, verbose=verbose)
        
        if self.teacher_model is None:
            teacher_ckpt = "/ultralytics/data/teacher_v0/best.pt"
            y = YOLO(teacher_ckpt)
            t_model = y.model
            for p in t_model.parameters():
                p.requires_grad = False
            t_model.eval()
            self.teacher_model = t_model
            print("[MoETrainer] teacher loaded:", teacher_ckpt)

            model.teacher_model = self.teacher_model
            print("[MoETrainer] teacher attached to model")

        return model