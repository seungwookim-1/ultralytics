import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect
from chimera import ChimeraYOLO

# 1. 기본 YOLOv11 모델 생성
base = DetectionModel(cfg="yolo11x.yaml", ch=3, nc=80, verbose=True)

# 2. 마지막 Detect 레이어 앞까지를 backbone으로 사용
#    (실제로는 어떤 레이어까지를 feature extractor로 쓸지 조금 더 정교하게 잘라야 함)
backbone = nn.Sequential(*list(base.model.children())[:-1])

# 3. 헤드 정의 예시
head_defs = {
    "area":    {"type": "detect", "nc": 5},
    "vehicle": {"type": "detect", "nc": 8},
    "weather": {"type": "cls",    "nc": 4},
}

heads = {}

# Detect head 예시 (실제로는 feature map 채널 수를 맞춰야 함)
heads["area"] = Detect(nc=head_defs["area"]["nc"], ch=base.model[-1].ch)
heads["vehicle"] = Detect(nc=head_defs["vehicle"]["nc"], ch=base.model[-1].ch)

# 날씨 분류 head 예시는 단일 feature map만 쓰는 식으로:
in_channels = base.model[-2].cv2.conv.out_channels  # 예: 가장 위 feature
heads["weather"] = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(in_channels, head_defs["weather"]["nc"])
)

lambdas = {"area": 1.0, "vehicle": 1.0, "weather": 0.5}

chimera = ChimeraYOLO(backbone=backbone, heads=heads, lambdas=lambdas)
