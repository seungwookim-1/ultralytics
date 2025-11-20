# chimera_core/dataset.py
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from SD_db_Library.dataloader import ImageLabelDataset

class DBYoloDataset(Dataset):
    def __init__(self, label_type: str = "area", transforms=None, split="train", split_ratio=0.1, seed=42):
        base = ImageLabelDataset(label_type=label_type)
        # base.image_df, base.label_df 이용해서 image/label 리스트 구성
        # + split(train/val) 로직도 여기서 처리

        self.samples = [...]   # (img_path, label_path or targets) 리스트
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, targets = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.transforms:
            img = self.transforms(img)

        return torch.from_numpy(img).permute(2, 0, 1), targets


# batch = {
#     "img": images.to(device),                      # [B, 3, H, W]
#     "targets": {
#         "road": road_targets.to(device),           # [N_road, 6]
#         "vehicle": vehicle_targets.to(device),     # [N_vehicle, 6]
#         # ...
#     }
# }