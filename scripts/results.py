from ultralytics import YOLO

moe = YOLO("/ultralytics/runs/detect/train24/weights/best.pt")
# results = moe("/ultralytics/outputs/")
expert_usage = moe.model.model[-1].expert_counts
print(f"Expert usage: {expert_usage}")