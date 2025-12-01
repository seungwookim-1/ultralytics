from single_symlink_path import create_dataset_config
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

if __name__ == "__main__":
    dataset_config_path = create_dataset_config(val_ratio=0.1, seed=42, max_train=1000, max_val=100)
    overrides = {
        "model": "yolo11-single.yaml",
        "data": str(dataset_config_path),
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.train()