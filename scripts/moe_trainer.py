from ultralytics import YOLO
from symlink_path import create_dataset_config

base = YOLO('yolo11n.pt')
# Load MoE model
moe = YOLO('yolo11-moe.yaml')
moe.model.load_state_dict(base.model.state_dict(), strict=False)

dataset_config_path = create_dataset_config(val_ratio=0.1, seed=42, max_train=1000, max_val=100)
# Train with MoE-specific settings

for k in ['scales', 'num_experts', 'backbone', 'head', 'nc', 'moe_aux_loss']:
    moe.overrides.pop(k, None)

results = moe.train(
    data= str(dataset_config_path),
    epochs=50,
)

# Inference (same as standard YOLO)
# results = moe('image.jpg')

# Analyze expert specialization
expert_usage = moe.model.model[-1].expert_counts
print(f"Expert usage: {expert_usage}")
