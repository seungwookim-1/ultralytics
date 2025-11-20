import os
from dotenv import load_dotenv

from ultralytics.nn.chimera_yolo import ChimeraYOLO
from chimera_core.dataset import DBYoloDataset
from chimera_core.trainer import train_loop

def main():
    # 1) .env 로드
    env_path = "/ultralytics/.env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print("Loaded .env")

    # 2) Multi-head 모델 구성
    head_defs = {
        "area":    {"type": "detect", "nc": 5},
        # "vehicle": {"type": "detect", "nc": 8},
        # "weather": {"type": "cls",    "nc": 4},
    }
    lambdas = {"area": 1.0}

    model = ChimeraYOLO(cfg="yolo11x.yaml", ch=3, nc=80, verbose=False, 
                        head_defs=head_defs, lambdas=lambdas)

    # 3) Dataset 로딩 (DB/NAS 기반)
    train_ds = DBYoloDataset(label_type="area", split="train", split_ratio=0.1, seed=42)
    # val_ds = DBYoloDataset(... split="val", ...)

    # 4) 학습 루프
    train_loop(model, train_ds, val_ds=None, epochs=100, batch_size=4, device="cuda")

if __name__ == "__main__":
    main()
