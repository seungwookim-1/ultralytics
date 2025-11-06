from ultralytics import YOLO
from ultralytics.models.yolo.detect.sd_trainer import ContinualDetectionTrainer

student = YOLO("yolo11n.pt")
teacher = YOLO("runs/vehicle/best.pt")  # 고정된 teacher

overrides = dict(
    model="yolo11n.yaml",
    data="coco8.yaml",
    epochs=20,
    imgsz=640,
    cfg="hyp_strong.yaml",
    project="runs",
    name="B_with_distill",
)

trainer = ContinualDetectionTrainer(overrides=overrides)

# 🔴 중요: 학생 모델을 "pretrained"로 로드
# 방법 1) student.model 객체 그대로 주입
trainer.model = trainer.get_model(cfg=overrides["model"], weights=None)
trainer.model.load(student.model)  # ← 이 줄이 핵심

# 방법 2) get_model 단계에서 바로 .pt를 주입하고 싶다면:
# trainer.model = trainer.get_model(cfg=overrides["model"], weights="yolo11n.pt")

trainer.set_teacher(teacher.model)

# (선택) 증류 하이퍼를 직접 세팅하는 방식이라면 여기서
trainer.lmb_feat = 0.5
trainer.lmb_kd   = 0.2
trainer.kd_T     = 2.0

trainer.train()
