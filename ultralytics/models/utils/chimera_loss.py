# import torch
# import torch.nn as nn

# from ultralytics.utils.loss import v8DetectionLoss
# from ultralytics.utils import LOGGER
# from ultralytics.utils.checks import check_yaml


# class _SingleHeadModelWrapper(nn.Module):
#     """
#     v8DetectionLoss가 기대하는 최소한의 인터페이스만 제공하는 래퍼.

#     - model.model[-1] -> 단일 Detect 헤드
#     - nc, args, stride 등은 원래 모델에서 그대로 가져오되,
#       nc는 이 헤드의 nc로 오버라이드
#     """
#     def __init__(self, full_model, detect_head, nc_head: int):
#         super().__init__()
#         self.model = nn.ModuleList(list(full_model.model))  # shallow copy
#         self.model[-1] = detect_head

#         # full_model이 가지고 있는 설정들 복사
#         self.args = getattr(full_model, "args", None)
#         self.stride = getattr(detect_head, "stride", None)
#         self.nc = nc_head

#         # device, dtype
#         self.to(next(full_model.parameters()).device)

#     def forward(self, x):
#         # 이 래퍼에서 forward는 쓰지 않고, loss에서 preds만 넣어줄 거라 pass
#         raise RuntimeError("SingleHeadModelWrapper.forward는 사용하지 않습니다.")


# class ChimeraDetectionLoss(nn.Module):
#     """
#     N-헤드 멀티헤드 Loss.

#     - preds: list[head_idx] -> Detect가 원래 반환하는 preds
#     - batch: YOLOv8 표준 배치 딕셔너리
#     - multi_heads_cfg: data.yaml의 multi_heads 딕셔너리
#     """

#     def __init__(self, model, multi_heads_cfg: dict, head_weights: dict | None = None):
#         super().__init__()
#         self.full_model = model
#         self.detect = model.model[-1]
#         self.multi_heads_cfg = multi_heads_cfg or {}
#         self.head_names = list(self.multi_heads_cfg.keys())

#         # 헤드 개수 sanity check
#         if len(self.head_names) != len(self.detect.heads):
#             LOGGER.warning(
#                 f"[ChimeraLoss] data.yaml의 multi_heads({len(self.head_names)})와 "
#                 f"모델 헤드 수({len(self.detect.heads)})가 다릅니다."
#             )

#         # 각 헤드별 글로벌→로컬 클래스 매핑 테이블 생성
#         # global_id in [0, model.nc) -> local_id in [0, nc_head) or -1 (해당 헤드가 안 보는 클래스)
#         self.class_maps = {}
#         model_nc = getattr(model, "nc", None) or getattr(model.model[-1], "nc", None)
#         if model_nc is None:
#             raise RuntimeError("[ChimeraLoss] 모델에서 nc를 찾을 수 없습니다.")

#         for idx, name in enumerate(self.head_names):
#             cfg = self.multi_heads_cfg[name]
#             global_ids = cfg["class_ids"]
#             local_map = torch.full((model_nc,), -1, dtype=torch.long)
#             for local_idx, gid in enumerate(global_ids):
#                 local_map[gid] = local_idx
#             self.class_maps[idx] = local_map  # head_idx 기준으로 저장

#         # 헤드별 기본 loss 객체 생성
#         self.head_losses: dict[int, v8DetectionLoss] = {}
#         for idx, name in enumerate(self.head_names):
#             cfg = self.multi_heads_cfg[name]
#             nc_head = len(cfg["class_ids"])
#             wrapper = _SingleHeadModelWrapper(model, self.detect.heads[idx], nc_head)
#             self.head_losses[idx] = v8DetectionLoss(wrapper)

#         # 헤드별 가중치
#         if head_weights is None:
#             # 디폴트: 전부 1.0
#             self.head_weights = {name: 1.0 for name in self.head_names}
#         else:
#             self.head_weights = head_weights

#     def _build_sub_batch(self, batch: dict, head_idx: int) -> dict:
#         """
#         YOLOv8 배치에서, 이 헤드가 관심 있는 클래스만 필터링해
#         서브 배치 생성.

#         가정:
#             batch['cls'] shape:
#               - (N, 1) : class index
#               - or (N, C): one-hot or multi-label
#         """
#         device = batch["img"].device
#         cls = batch["cls"].to(device)
#         bboxes = batch["bboxes"].to(device)
#         batch_idx = batch["batch_idx"].to(device)

#         class_map = self.class_maps[head_idx].to(device)

#         if cls.ndim == 2 and cls.shape[1] == 1:
#             # (N, 1) : index 포맷
#             global_ids = cls.squeeze(1).long()  # (N,)
#         else:
#             # (N, C): one-hot/multi-label → argmax로 index 근사
#             global_ids = cls.argmax(dim=1)

#         local_ids = class_map[global_ids]  # (N,)

#         # 이 헤드가 보지 않는 클래스(-1)는 제거
#         keep_mask = local_ids >= 0
#         if keep_mask.sum() == 0:
#             # 이 배치에서 이 헤드가 학습할 게 없으면, 빈 배치 구성
#             sub_batch = {k: v for k, v in batch.items()}
#             # YOLOv8은 cls/bboxes/batch_idx가 empty여도 동작하도록 설계되어 있음
#             sub_batch["cls"] = cls[:0]
#             sub_batch["bboxes"] = bboxes[:0]
#             sub_batch["batch_idx"] = batch_idx[:0]
#             return sub_batch

#         sub_batch = {k: v for k, v in batch.items()}
#         sub_batch["cls"] = local_ids[keep_mask].unsqueeze(1).float()  # (M, 1)
#         sub_batch["bboxes"] = bboxes[keep_mask]
#         sub_batch["batch_idx"] = batch_idx[keep_mask]
#         return sub_batch

#     def forward(self, preds, batch):
#         """
#         preds: list[head_idx] -> Detect가 반환한 preds
#         batch: 원 배치 dict
#         """
#         total_loss = 0.0
#         # Ultralytics convention: [box, cls, dfl, ...]
#         # 여기서는 헤드별로 라벨을 붙여서 반환
#         loss_items = {}

#         for head_idx, name in enumerate(self.head_names):
#             if head_idx >= len(preds):
#                 continue  # 안전 장치

#             sub_batch = self._build_sub_batch(batch, head_idx)
#             head_loss, head_items = self.head_losses[head_idx](preds[head_idx], sub_batch)
#             w = self.head_weights.get(name, 1.0)

#             total_loss = total_loss + w * head_loss

#             # head_items는 [box, cls, dfl, ...] 텐서로 가정
#             loss_items[f"{name}/box_loss"] = head_items[0] * w
#             loss_items[f"{name}/cls_loss"] = head_items[1] * w
#             loss_items[f"{name}/dfl_loss"] = head_items[2] * w

#         return total_loss, loss_items
