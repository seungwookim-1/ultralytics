from ultralytics import YOLO
from symlink_path import create_dataset_config

# 1) dataset 고정
dataset_config_path = create_dataset_config(
    val_ratio=0.1,
    seed=42,
    max_train=1000,
    max_val=100,
)

# 2) baseline / MoE 모델 생성
base = YOLO("yolo11n.pt")
moe  = YOLO("yolo11-moe.yaml")

# 3) baseline backbone을 MoE backbone에만 이식 (head는 그대로 사용)
for name, p in base.model.named_parameters():
    if name.startswith("model.0"):  # backbone stem prefix는 실제 구조에 맞춰 수정
        moe.model.state_dict()[name].copy_(p)

# 4) 공통 학습 설정
common_hp = dict(
    epochs=50,
    lr0=0.01,
    warmup_epochs=3,
    seed=42,
    batch=16,
    imgsz=640,
)

# 5) train
results_base = base.train(data=str(dataset_config_path), **common_hp)
results_moe  = moe.train(data=str(dataset_config_path), **common_hp)

# 6) expert usage
print("Expert usage:", moe.model.model[-1].expert_counts)
