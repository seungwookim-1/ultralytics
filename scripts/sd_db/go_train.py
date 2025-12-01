# from dotenv import load_dotenv
# load_dotenv('/ultralytics/.env', override=True)

from ultralytics import YOLO
from pathlib import Path
from filter_data_path import create_dataset_config

# Import SD_db_Library for database operations
print("Starting YOLOv11 training with SD_db_Library logging...")

# Create dataset YAML with random val split. Default val_ratio=0.1 & seed=42
dataset_config_path = create_dataset_config(data_conditions={"label_type": "area"})
model = YOLO("yolo11x.yaml")

train_config = {
    'data': str(dataset_config_path),
    'epochs': 50,
    'imgsz': 1280,
    'batch': 4,
    'fliplr': 0,
    'project': '/outputs',
    'name': 'train-1280-3'
}

model.train(**train_config)

# Log training completion if DB is available
completion_data = {
            'status': 'completed',
            'final_model_path': f"{train_config['project']}/{train_config['name']}/weights/best.pt"
        }
print(f"Training completed: {completion_data}")
