import torch
import torch.nn as nn

class ChimeraYOLO(nn.Module):
    def __init__(self, backbone: nn.Module,
                 heads: dict[str, nn.Module],
                 lambdas: dict[str, float] | None = None):
        """
        backbone: YOLOv11 backbone(+FPN) 모듈
        heads:   {"road": head_road, "vehicle": head_vehicle, "weather": head_weather, ...}
        lambdas: {"road": 1.0, "vehicle": 1.0, "weather": 0.5, ...}  # loss weight
        """
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.lambdas = lambdas or {}

    def forward(self, x, targets: dict | None = None):
        """
        x: [B, C, H, W]
        targets:
            {"road": road_targets, "vehicle": vehicle_targets, ...}
            - 형식은 우리가 정의 (예: YOLO 포맷, cls+box 텐서 등)

        eval 모드: {head_name: preds} 딕셔너리만 반환
        train 모드: {"total_loss": ..., "per_head": {...}, "preds": {...}}
        """
        # 1) 공유 백본
        feats = self.backbone(x)   # 예: [P3, P4, P5] 같은 리스트 or 단일 텐서

        # 2) 헤드별 forward
        preds = {}
        for name, head in self.heads.items():
            preds[name] = head(feats)

        # 3) 평가 모드 → 예측만
        if (not self.training) or (targets is None):
            return preds

        # 4) 학습 모드 → head별 loss + total_loss
        total_loss = 0.0
        per_head = {}

        for name, pred in preds.items():
            if name not in targets:
                continue
            loss = self._compute_head_loss(name, pred, targets[name])
            w = self.lambdas.get(name, 1.0)
            total_loss = total_loss + w * loss
            per_head[name] = loss.detach()

        return {
            "total_loss": total_loss,
            "per_head": per_head,
            "preds": preds,
        }

    def _compute_head_loss(self, name, pred, target):
        """
        head별 loss 정의 위치:
        - detect head면 YOLO-style loss
        - weather head면 cross entropy
        - etc...
        """
        raise NotImplementedError
