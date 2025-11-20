# ultralytics/nn/chimera_yolo.py

import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect   # 현재 YOLOv11 Detect 모듈
from ultralytics.nn.tasks import DetectionModel

class MultiHeadDetect(nn.Module):
    def __init__(self, base_detect: Detect, head_defs: dict, lambdas: dict | None = None):
        """
        base_detect: 기존 DetectionModel 의 마지막 Detect 모듈
        head_defs:   {"road": {"nc":5}, "vehicle":{"nc":8}, ...}
        lambdas:     {"road":1.0, "vehicle":0.5, ...}
        """
        super().__init__()
        self.stride = base_detect.stride
        self.heads = nn.ModuleDict()
        self.lambdas = lambdas or {}

        for name, cfg in head_defs.items():
            nc = cfg["nc"]
            head = Detect(
                nc=nc,
                ch=base_detect.ch,
            )
            # base_detect의 weight 초기값을 복사 (선택적, 수렴 안정화)
            # Note: Detect head는 cv2, cv3 ModuleList를 가지므로 직접 복사는 복잡함
            # 대신 bias_init()이 자동으로 호출됨
            self.heads[name] = head

    def forward(self, x):
        """
        x: [P3, P4, P5] 형태의 feature 리스트 (기존 Detect 와 동일)
        return: {"road": pred_road, "vehicle": pred_vehicle, ...}
        """
        out = {}
        for name, head in self.heads.items():
            out[name] = head(x)
        return out

class ChimeraYOLO(DetectionModel):
    """
    기존 YOLOv11 DetectionModel + MultiHeadDetect
    """
    def __init__(self, cfg="yolo11x.yaml", ch=3, nc=None, verbose=True,
                 head_defs: dict | None = None, lambdas: dict | None = None):
        # 우선 부모 DetectionModel 을 정상적으로 초기화
        super().__init__(cfg, ch=ch, nc=nc or 80, verbose=verbose)

        if head_defs is None:
            raise ValueError("head_defs must be provided for ChimeraYOLO")

        self.head_defs = head_defs
        self.lambdas = lambdas or {}

        # 마지막 모듈이 Detect 인지 확인
        base_detect = self.model[-1]
        assert isinstance(base_detect, Detect), "Last layer must be Detect"

        # Detect 를 MultiHeadDetect 로 교체
        mh = MultiHeadDetect(base_detect, head_defs, lambdas)
        self.model[-1] = mh

        # stride 공유
        self.stride = mh.stride

        # YOLO 기본 loss 초기화
        # v8DetectionLoss는 model에서 nc를 읽으므로, 각 head에 맞는 임시 모델 생성
        from ultralytics.utils.loss import v8DetectionLoss
        
        self.criterion_per_head = {}
        for name, head_cfg in head_defs.items():
            if head_cfg.get("type") == "detect":
                # 각 head에 대한 loss를 위해 임시 DetectionModel 생성
                # 이 모델은 loss 초기화에만 사용되고 실제 학습에는 사용되지 않음
                # cfg는 모델 config 파일 경로, head_cfg["nc"]는 클래스 수
                temp_model = DetectionModel(cfg, ch=ch, nc=head_cfg["nc"], verbose=False)
                self.criterion_per_head[name] = v8DetectionLoss(temp_model)

    def forward(self, x, targets: dict | None = None):
        """
        x: 이미지 텐서 [B, 3, H, W] 또는 dict (BaseModel 호환)
        targets: {"area": batch_dict, "vehicle": batch_dict, ...}
                 각 batch_dict는 {"batch_idx": tensor, "cls": tensor, "bboxes": tensor, "img": tensor} 형식
                 또는 targets가 None이면 inference 모드
        """
        # BaseModel의 forward는 dict를 받으면 loss를 호출하므로, 여기서는 직접 처리
        if isinstance(x, dict):
            # BaseModel 호환: dict를 받으면 loss 호출
            return self.loss(x)
        
        # BaseModel의 forward 로직을 따라가되, MultiHeadDetect가 dict를 반환하도록 처리
        y = []  # save list
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if getattr(m, "i", -1) in self.save else None)

        preds_per_head = x  # MultiHeadDetect가 dict를 반환: {"area": [P3, P4, P5], ...}

        if self.training and targets is not None:
            return self._forward_train(preds_per_head, targets)
        else:
            return preds_per_head

    def _forward_train(self, preds_per_head, targets_per_head: dict):
        """
        preds_per_head: {"area": [P3, P4, P5], ...} - 각 head의 training predictions
        targets_per_head: {"area": {"batch_idx": ..., "cls": ..., "bboxes": ..., "img": ...}, ...}
        """
        total_loss = 0.0
        head_losses = {}

        for name, preds in preds_per_head.items():
            if name not in targets_per_head or name not in self.criterion_per_head:
                continue
            
            batch = targets_per_head[name]
            # batch는 dict 형태여야 함: {"batch_idx": tensor, "cls": tensor, "bboxes": tensor, ...}
            
            # YOLO detection loss 호출
            # v8DetectionLoss는 (preds, batch)를 받고 (total_loss, loss_components)를 반환
            loss_total, loss_components = self.criterion_per_head[name](preds, batch)
            
            w = self.lambdas.get(name, 1.0)
            total_loss += w * loss_total
            head_losses[name] = {
                "total": loss_total.detach().item(),
                "components": [x.item() for x in loss_components] if isinstance(loss_components, torch.Tensor) else loss_components
            }

        return {
            "total_loss": total_loss,
            "per_head": head_losses,
        }
    
    def loss(self, batch, preds=None):
        """Compute loss for multi-head model.
        
        Args:
            batch: Dict with structure {"img": tensor, "targets": {"area": batch_dict, ...}}
                   where each batch_dict has {"batch_idx": tensor, "cls": tensor, "bboxes": tensor}
            preds: Optional precomputed predictions
        
        Returns:
            tuple: (total_loss, loss_dict) for compatibility with BaseModel
        """
        if preds is None:
            # Forward pass to get predictions
            preds = self.forward(batch["img"], targets=batch.get("targets"))
        
        if isinstance(preds, dict) and "total_loss" in preds:
            # Already computed loss in forward
            total_loss = preds["total_loss"]
            # Create loss components dict for compatibility
            loss_dict = preds.get("per_head", {})
            return total_loss, loss_dict
        else:
            # If preds is just predictions, compute loss
            return self._forward_train(preds, batch.get("targets", {}))
