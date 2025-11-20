from ultralytics import YOLO
from ultralytics.models.yolo.detect.sd_trainer import ContinualDetectionTrainer
from pathlib import Path
from ultralytics.data.utils import check_det_dataset


W = Path("/ultralytics/yolo11n.pt")
assert W.exists(), f"weights not found: {W}"

teacher = YOLO("runs/A_teacher/weights/best.pt")
student = YOLO(str(W))

overrides = dict(
    model="yolo11n.yaml",
    data="coco8.yaml",
    epochs=5,
    imgsz=640,
    #cfg="hyp_strong.yaml",   # 증강/하이퍼
    project="runs",
    # name="B_with_distill",
    name="B_with_kd005"
)

trainer = ContinualDetectionTrainer(overrides=overrides)
# trainer.model = trainer.get_model(cfg=overrides["model"], weights=str(W))
# trainer.model.nc = trainer.data["nc"]
# trainer.model.names = trainer.data["names"]


# trainer.set_teacher(teacher.model)

# trainer.lmb_feat = 0.0
# trainer.lmb_kd   = 0.0
# trainer.kd_T     = 2.0

# trainer.train()

trainer.model = student.model
trainer.model.train()  # 학습 모드 보장

# 3) 데이터-헤드 정합 고정
trainer.model.nc = trainer.data["nc"]
trainer.model.names = trainer.data["names"]


# ✅ split 오염 제거 + datadict 강제 보정
trainer.args.split = None          # ★ 가장 중요
trainer.args.val = True            # (원하면 False로 꺼도 됨)
trainer.data = check_det_dataset(trainer.args.data)  # train/val 경로 재확인

# ✅ 첫 배치 강제 로드 (길이가 0이면 바로 터짐)
dl = trainer.get_dataloader(trainer.data["train"], batch_size=2, rank=-1, mode="train")
it = iter(dl)
first = next(it)
print("[DBG] first batch:", first["img"].shape, "n_labels:", sum(len(x) for x in first["cls"]))


# 4) teacher 세팅
trainer.set_teacher(teacher.model)

# 5) 우선 증류 완전 OFF로 파이프라인 정상성 확인
trainer.lmb_feat = 0.0
trainer.lmb_kd   = 0.05
trainer.kd_T     = 2.0

trainer.args.epochs = 5

trainer.train()
