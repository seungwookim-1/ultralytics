from ultralytics import YOLO
from ultralytics.models.yolo.detect.sd_trainer import ContinualDetectionTrainer

teacher = YOLO("runs/A_teacher/weights/best.pt")
student = YOLO("yolo11n.pt")

overrides = dict(
    model="yolo11n.yaml",
    data="coco8.yaml",
    epochs=20,
    imgsz=640,
    cfg="hyp_strong.yaml",   # 증강/하이퍼는 여기로
    project="runs",
    name="B_with_distill",
)

trainer = ContinualDetectionTrainer(overrides=overrides)
trainer.model = trainer.get_model(cfg=overrides["model"], weights=None)
trainer.set_teacher(teacher.model)

trainer.lmb_feat = 0.5
trainer.lmb_kd   = 0.2
trainer.kd_T     = 2.0

trainer.train()