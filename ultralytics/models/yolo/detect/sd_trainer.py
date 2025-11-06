from ultralytics.models.yolo import detect

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import DetectionModel


def feat_distill(feats_new: dict, feats_old: dict) -> torch.Tensor:
    loss = 0.0
    for k in ("P3", "P4", "P5"):
        if k in feats_new and k in feats_old and feats_new[k] is not None and feats_old[k] is not None:
            loss = loss + (feats_new[k] - feats_old[k]).pow(2).mean()
    return loss

def logit_distill(logits_new, logits_old, T: float = 2.0) -> torch.Tensor:
    # logits_*: [N, C, ...] → detect head의 class/logit에 맞게 꺼낸 텐서를 넣으세요.
    pn = F.log_softmax(logits_new / T, dim=1)
    po = F.softmax(logits_old / T, dim=1)
    return F.kl_div(pn, po, reduction='batchmean') * (T * T)

class ContinualDetectionTrainer(detect.DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # self.model 은 student
        self.teacher = None
        # 하이퍼 파라미터(증류 람다 등)는 overrides로 받기
        self.lmb_feat = float(self.args.get("lambda_feat", 0.5))
        self.lmb_kd   = float(self.args.get("lambda_kd", 0.2))
        self.kd_T     = float(self.args.get("kd_T", 2.0))

    def set_teacher(self, teacher_model: DetectionModel):
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def train_step(self, batch):
        batch = self.preprocess_batch(batch)
        imgs, labels = batch["img"], batch["labels"]

        # 1) student forward
        feats_new = self.model.forward_features(imgs, normalize=True)   # << 여기서 중간피처
        preds_new = self.model(imgs)                                    # 기존 예측
        loss, loss_items = self.criterion(preds_new, batch)             # 기본 손실 (box/cls/dfl)

        # 2) teacher distillation
        if self.teacher is not None and (self.lmb_feat > 0.0 or self.lmb_kd > 0.0):
            with torch.no_grad():
                feats_old = self.teacher.forward_features(imgs, normalize=True)
                preds_old = self.teacher(imgs)

            if self.lmb_feat > 0.0:
                loss = loss + self.lmb_feat * feat_distill(feats_new, feats_old)

            if self.lmb_kd > 0.0:
                # preds_*에서 KD 대상(logit)을 꺼내는 헬퍼가 필요할 수 있음
                # 예: preds["det_logits"] 처럼 꺼내도록 DetectionModel 예측 포맷에 맞춰 수정
                kd_new, kd_old = extract_det_logits(preds_new), extract_det_logits(preds_old)
                loss = loss + self.lmb_kd * logit_distill(kd_new, kd_old, self.kd_T)

        return loss, loss_items

def _get_detect_head(model: nn.Module):
    # DDP 래핑이 있으면 벗겨서 마지막 모듈(Detect)을 얻음
    m = model
    if hasattr(m, "module"):
        m = m.module
    head = m.model[-1]
    return head

def extract_det_logits(preds, model: nn.Module) -> torch.Tensor:
    """
    preds: Detect head의 순전파 출력(스케일별 리스트/튜플, 혹은 dict/단일 텐서)
    model: student 또는 teacher 모델(클래스 수 등 메타 읽기용)

    return: [B, N, C]  (N=모든 스케일의 grid*anchors 평탄화, C=nc)
    """
    detect = _get_detect_head(model)
    nc = int(getattr(detect, "nc", 0))
    if nc <= 0:
        raise ValueError("Detect head 'nc' not found or zero.")

    # preds를 리스트 형태로 정규화
    if isinstance(preds, dict) and "pred" in preds:
        outs = preds["pred"]
    elif isinstance(preds, (list, tuple)):
        outs = preds
    else:
        outs = [preds]

    cls_list = []
    for p in outs:
        # 스케일별 텐서 p를 클래스 로짓만 추출
        if p.dim() == 4:
            # [B, C, H, W] 형태 가정 → 채널 끝 nc가 클래스
            # (YOLOv8/11 Detect는 reg/DFL 뒤에 cls가 붙음)
            if p.size(1) < nc:
                raise ValueError(f"Channel dim {p.size(1)} < nc {nc}")
            cls_map = p[:, -nc:, ...]                  # [B, nc, H, W]
            cls_flat = cls_map.permute(0, 2, 3, 1).reshape(p.size(0), -1, nc)  # [B, H*W, nc]
        elif p.dim() == 5:
            # [B, A, H, W, no] 형태 → 마지막 dim의 뒤 nc가 클래스
            if p.size(-1) < nc:
                raise ValueError(f"no {p.size(-1)} < nc {nc}")
            cls_flat = p[..., -nc:].reshape(p.size(0), -1, nc)                # [B, A*H*W, nc]
        else:
            raise ValueError(f"Unsupported preds dim: {p.dim()}")

        cls_list.append(cls_flat)

    if not cls_list:
        raise ValueError("No class logits extracted.")

    return torch.cat(cls_list, dim=1)  # [B, N_total, nc]