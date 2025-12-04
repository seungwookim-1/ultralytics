from ultralytics import YOLO
from ultralytics.nn.modules.head import MoEDetect
from custom_core.symlink_helper import create_dataset_config
from custom_core.moe_trainer import MoETrainer

# 1) dataset 고정
dataset_config_path = create_dataset_config(
    val_ratio=0.1, seed=42, max_train=1000, max_val=100
)

# 2) baseline, MoE 모델 로드
base = YOLO("/ultralytics/data/teacher_v0/best.pt")          # moe와 같은 teacher 사용
moe  = YOLO("yolo11-moe.yaml")     # 구조만 정의된 MoE

# 3) ★ backbone 포함 전체 가중치를 MoE로 복사 (strict=False)
moe.model.load_state_dict(base.model.state_dict(), strict=False)

# 4) Detect head → MoE head 복사
base_head = base.model.model[-1]
moe_head  = moe.model.model[-1]
assert isinstance(moe_head, MoEDetect)
moe_head.init_from_detect(base_head, noise_scale=0.01) 

moe.ckpt = True

# 5) 공통 하이퍼
common_config = dict(
    data=str(dataset_config_path),
    epochs=50,
    batch=16,
    imgsz=640,
    seed=42,
)

# 6) MoE 학습
results_moe = moe.train(
    trainer = MoETrainer,
    # teacher_ckpt="/ultralytics/data/teacher_v0/best.pt",
    **common_config,
)
print("Expert usage:", moe_head.expert_counts)

# 7) 비교군 (baseline) 학습
results_base = base.train(
    **common_config,
)

print("Expert usage:", moe_head.expert_counts)
