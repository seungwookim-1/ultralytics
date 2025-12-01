from ultralytics import YOLO

# Load MoE model
model = YOLO('yolo11-moe.yaml')

# Train with MoE-specific settings
results = model.train(
    data='coco.yaml',
    epochs=100,
    # imgsz=640,
    # batch=16,  # May need to reduce
    moe_aux_loss=0.01,  # Auxiliary loss weight
)

# Inference (same as standard YOLO)
results = model('image.jpg')

# Analyze expert specialization
expert_usage = model.model.model[-1].expert_counts
print(f"Expert usage: {expert_usage}")
