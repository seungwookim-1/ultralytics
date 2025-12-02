from ultralytics import YOLO
from ultralytics.nn.modules.head import MoEDetect
from symlink_helper import create_dataset_config
from moe_trainer import MoETrainer


# 1) dataset 고정
dataset_config_path = create_dataset_config(
    val_ratio=0.1, seed=42, max_train=1000, max_val=100
)

# 2) baseline, MoE 모델 로드
base = YOLO("yolo11n.pt")          # COCO pretrain 완료 모델
moe  = YOLO("yolo11-moe.yaml")     # 구조만 정의된 MoE

# 3) ★ backbone 포함 전체 가중치를 MoE로 복사 (strict=False)
moe.model.load_state_dict(base.model.state_dict(), strict=False)

# 4) Detect head → MoE head 복사
base_head = base.model.model[-1]
moe_head  = moe.model.model[-1]
assert isinstance(moe_head, MoEDetect)
moe_head.init_from_detect(base_head, noise_scale=0.01) 

moe.ckpt = True

common_hp = dict(
    epochs=50,
    batch=16,
    imgsz=640,
    seed=42,
)

# 7) MoE 학습

results_moe = moe.train(
    data=str(dataset_config_path),
    trainer=MoETrainer,   # 여기서 teacher를 model에 붙여주기만 함
    # moe_kd_weight=1.0,    # model.args로 들어가서 MoEDetectionLoss가 읽음
    # moe_kd_temp=1.0,
    **common_hp,
)
# Inference (same as standard YOLO)
# results = moe('image.jpg')

# Analyze expert specialization
print("Expert usage:", moe_head.expert_counts)
