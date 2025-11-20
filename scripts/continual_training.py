from ultralytics import YOLO
from ultralytics.models.yolo.detect.sd_trainer import ContinualDetectionTrainer
from ultralytics.data.utils import check_det_dataset

# 0) 초기 student 로드(혹은 직전 스테이지의 best.pt)
student = YOLO("/ultralytics/yolo11n.pt")  # 시작점

# 스테이지 정의: (teacher_w, data_yaml, epochs, kd_lambda)
STAGES = [
    ("runs/A_teacher/weights/best.pt", "data/coco8.yaml",        5, 0.05),
    ("runs/B_teacher/weights/best.pt", "data/your_task_v1.yaml", 5, 0.05),
    ("runs/C_teacher/weights/best.pt", "data/your_task_v2.yaml", 8, 0.10),
]

for si, (teacher_w, data_yaml, epochs, l_kd) in enumerate(STAGES, start=1):
    overrides = dict(
        model=None,            # 외부에서 student.model 주입
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        project="runs",
        name=f"KD_stage{si}",
        # 필요 시 증강/하이퍼: cfg="hyp_strong.yaml",
        # EMA를 쓰면 스테이지 간 일반화가 더 부드러워짐
        ema=True,
    )

    trainer = ContinualDetectionTrainer(overrides=overrides)
    trainer.data = check_det_dataset(trainer.args.data)

    # --- 학생 이어붙이기 (누적의 핵심) ---
    trainer.model = student.model
    trainer.model.train()
    trainer.model.nc = trainer.data["nc"]
    trainer.model.names = trainer.data["names"]

    # --- 교사 세팅 ---
    teacher = YOLO(teacher_w)
    trainer.set_teacher(teacher.model)

    # --- KD 하이퍼 ---
    trainer.lmb_feat = 0.0
    trainer.lmb_kd   = l_kd
    trainer.kd_T     = 2.0

    # 학습
    trainer.train()

    # 이 스테이지 결과(best)를 다음 스테이지 시작점으로 삼음
    # Ultralytics는 trainer가 best를 저장하니 이를 다시 불러 학생 갱신
    student = YOLO(f"runs/KD_stage{si}/weights/best.pt")
