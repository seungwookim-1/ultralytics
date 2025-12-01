import torch
import torch.nn as nn
import numpy as np
from copy import copy

from ultralytics.nn.tasks import ChimeraDetectionModel
from ultralytics.utils import DEFAULT_CFG, RANK, ops
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import box_iou
# from chimera import ChimeraYOLO

# 1. 기본 YOLOv11 모델 생성
class ChimeraDetectionTrainer(DetectionTrainer):
    """
    ChimeraDetectionModel을 사용하는 트레이너.

    - get_model()만 ChimeraDetectionModel로 바꿔서 사용
    - 나머지 build_dataset, preprocess_batch는 DetectionTrainer 그대로 사용 가능
      (단, dataset이 batch["nonmoving"], batch["rider"]를 만들어줘야 함)
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # 1) overrides는 dict 여야 한다
        overrides = dict(overrides or {})

        # 2) custom 키(lambdas)는 여기서 미리 빼두고,
        #    base Trainer에게는 넘기지 않아야 cfg 검증이 안 깨진다.
        self.lambdas = overrides.pop("lambdas", None)

        # 4) model이 없으면 기본 멀티헤드 yaml 사용
        overrides.setdefault("model", "yolo11-chimera.yaml")

        # 5) 나머지는 기본 DetectionTrainer 로 위임
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        # cfg가 None이면 args.model(yaml)을 사용
        cfg = cfg or self.args.model

        model = ChimeraDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],              # 전체 class 수(실제로는 크게 중요치 않을 수 있음)
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            lambdas=getattr(self.args, "lambdas", None),
        )
        model.data = self.data
        if weights:
            model.load(weights)
        return model


    def get_validator(self):
        """
        ChimeraDetectionModel + ChimeraDetectionValidator 조합 사용.
        """
        # 원래 DetectionTrainer.get_validator() 내부 구현을 참고해서,
        # 여기서는 Validator 클래스만 교체
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")

        validator = ChimeraDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
        )

        validator.data = self.data
        validator.model = self.model

        # Ensure the model has access to data for multi-head mapping
        if not hasattr(self.model, "data") or self.model.data is None:
            self.model.data = self.data

        return validator


def _bbox_iou(box1, box2):
    """
    box1: (..., 4), box2: (N, 4)  # xyxy
    return: (..., N) IoU
    """
    # (x1,y1,x2,y2)
    x1 = torch.max(box1[..., 0], box2[:, 0])
    y1 = torch.max(box1[..., 1], box2[:, 1])
    x2 = torch.min(box1[..., 2], box2[:, 2])
    y2 = torch.min(box1[..., 3], box2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    union = area1 + area2 - inter + 1e-6
    return inter / union


def _nms_single_image(
    boxes_xyxy: torch.Tensor,
    conf: torch.Tensor,
    cls: torch.Tensor,
    iou_thres: float,
) -> torch.Tensor:
    """
    간단한 per-image NMS.

    - boxes_xyxy: (N, 4)  [x1, y1, x2, y2]
    - conf      : (N,) 혹은 (N, 1)  → 내부에서 (N,) 으로 펴줌
    - cls       : (N,)   (지금은 클래스별 NMS 안 하고, 전체에서만 NMS)
    - 반환: keep 인덱스 (LongTensor, shape (M,))
    """
    device = boxes_xyxy.device

    # 0) 모양 강제 정리 (여기서부터는 무조건 1D/2D 통일)
    boxes_xyxy = boxes_xyxy.reshape(-1, 4)  # (N,4)
    conf = conf.reshape(-1)                 # (N,)
    cls = cls.reshape(-1)                   # (N,)

    N = boxes_xyxy.shape[0]
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # 1) conf 기준 내림차순 정렬 → idxs는 무조건 1D로 flatten
    idxs = torch.argsort(conf, descending=True).reshape(-1)  # (N,)

    # --- 디버그: 한 번만 찍어서 idxs 모양 확인 ---
    if not hasattr(_nms_single_image, "_debug_once"):
        _nms_single_image._debug_once = True
        print("[NMS DEBUG] boxes_xyxy.shape =", boxes_xyxy.shape)
        print("[NMS DEBUG] conf.shape       =", conf.shape)
        print("[NMS DEBUG] cls.shape        =", cls.shape)
        print("[NMS DEBUG] idxs.shape       =", idxs.shape)

    keep = []

    # 2) 전형적인 NMS 루프
    while idxs.numel() > 0:
        # 혹시라도 idxs가 2D가 되어 있으면 여기서 다시 1D로 강제
        idxs = idxs.reshape(-1)

        # 이제 idxs[0]은 0-dim 텐서라 .item() 가능
        i = int(idxs[0].item())
        keep.append(i)

        if idxs.numel() == 1:
            break

        cur_box = boxes_xyxy[i].unsqueeze(0)  # (1,4)

        other_idxs = idxs[1:].reshape(-1)     # (M,)
        other_boxes = boxes_xyxy[other_idxs]  # (M,4)

        ious = box_iou(cur_box, other_boxes)[0]  # (M,)

        remain_mask = ious <= iou_thres          # (M,)
        # remain_mask 도 1D 이므로, 결과도 항상 1D
        idxs = other_idxs[remain_mask]

    return torch.tensor(keep, dtype=torch.long, device=device)


class ChimeraDetectionValidator(DetectionValidator):
    def init_metrics(self, model):
        """Initialize metrics and ensure model has data for multi-head mapping."""
        super().init_metrics(model)

        # Ensure ChimeraDetectionModel has access to data for multi-head class mapping
        if hasattr(model, '__class__') and 'Chimera' in model.__class__.__name__:
            if not hasattr(model, "data") or model.data is None:
                model.data = self.data

    def postprocess(self, preds):
        # 1) ChimeraDetection.predict → (B, A, 4+53)
        if isinstance(preds, torch.Tensor):
            B, A, C = preds.shape
            box_ch = 4
            nc_total = C - box_ch  # 53
            device = preds.device

            box_xywh = preds[..., :4]       # (B, A, 4)
            box_xyxy = ops.xywh2xyxy(box_xywh)
            cls_logits = preds[..., 4:]     # (B, A, 53)

            # sigmoid → prob
            cls_prob = cls_logits.sigmoid()                 # (B, A, 53)
            obj = cls_prob.max(dim=-1, keepdim=True).values # (B, A, 1)

            # conf = obj * cls_prob
            conf_all = obj * cls_prob                       # (B, A, 53)
            conf_max, cls_ids = conf_all.max(dim=-1)        # (B, A), (B, A)

            outputs = []
            conf_thres = float(self.args.conf)
            iou_thres = float(self.args.iou)

            for b in range(B):
                boxes_b = box_xyxy[b]      # (A, 4), torch
                conf_b = conf_max[b]       # (A,), torch
                cls_b = cls_ids[b].float() # (A,), torch

                # 1) conf threshold
                mask = conf_b > conf_thres
                if mask.sum() == 0:
                    outputs.append(
                        {
                            "bboxes": torch.zeros((0, 4), device=device),
                            "conf": torch.zeros((0,), device=device),
                            "cls": torch.zeros((0,), device=device),
                        }
                    )
                    continue

                boxes_b = boxes_b[mask]    # (N, 4) - already in xyxy format
                conf_b = conf_b[mask]      # (N,)
                cls_b = cls_b[mask]        # (N,)

                # boxes_b is already in xyxy format from line 177, no conversion needed
                boxes_xyxy = boxes_b

                if boxes_xyxy.numel() == 0:
                    outputs.append(
                        {
                            "bboxes": torch.zeros((0, 4), device=device),
                            "conf": torch.zeros((0,), device=device),
                            "cls": torch.zeros((0,), device=device),
                        }
                    )
                    continue

                # 3) NMS (torch 기반)
                keep = _nms_single_image(boxes_xyxy, conf_b, cls_b, iou_thres)

                if keep.numel() == 0:
                    outputs.append(
                        {
                            "bboxes": torch.zeros((0, 4), device=device),
                            "conf": torch.zeros((0,), device=device),
                            "cls": torch.zeros((0,), device=device),
                        }
                    )
                    continue

                boxes_kept = boxes_xyxy[keep]  # torch (M, 4)
                conf_kept = conf_b[keep]       # torch (M,)
                cls_kept = cls_b[keep]         # torch (M,)

                outputs.append(
                    {
                        "bboxes": boxes_kept,
                        "conf": conf_kept,
                        "cls": cls_kept,
                    }
                )

            # 🔥 이 상태면 val.py의 타입 기대와 완전히 호환됨
            return outputs

        # 그 외에는 원래 YOLO 경로 사용
        return super().postprocess(preds)
