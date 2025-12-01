# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import enum
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
    ChimeraDetect
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML, colorstr, emojis, ops
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)


class BaseModel(torch.nn.Module):
    """Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion, information
    display, and weight loading capabilities.

    Attributes:
        model (torch.nn.Module): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.

    Methods:
        forward: Perform forward pass for training or inference.
        predict: Perform inference on input tensor.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimization.
        info: Print model information.
        load: Load weights into the model.
        loss: Compute loss for training.

    Examples:
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """

    def forward(self, x, *args, **kwargs):
        """Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # remove one2many head
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image that the model will be trained on.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(
            m, Detect
        ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented
    inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        """Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, YOLOESegment, Pose, OBB)) else self.forward(x)

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int): Flip type (0=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails.

        Args:
            y (list[torch.Tensor]): List of detection tensors.

        Returns:
            (list[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)

class _HeadLossProxy(nn.Module):
    """
    단일 Detect 헤드를 v8DetectionLoss가 기대하는 형태(BaseModel 비슷한 인터페이스)로 감싸는 래퍼.
    """
    def __init__(self, detect_head: nn.Module, args):
        super().__init__()
        # 1) Detect 모듈만 들고 있는 작은 model 처럼 구성
        self.model = nn.ModuleList([detect_head])
        self.args = args

        # 2) 🔥 stride가 0이면, nl 기준으로 합리적인 기본 stride를 넣어준다.
        m = self.model[-1]
        if hasattr(m, "stride"):
            stride = m.stride
            if isinstance(stride, torch.Tensor):
                if (stride == 0).all():
                    nl = getattr(m, "nl", 3)
                    base = 8.0
                    vals = [base * (2 ** i) for i in range(nl)]  # [8, 16, 32, ...]
                    m.stride = torch.tensor(vals, device=stride.device)
            else:
                try:
                    if all(s == 0 for s in stride):
                        nl = getattr(m, "nl", len(stride))
                        base = 8.0
                        vals = [base * (2 ** i) for i in range(nl)]
                        m.stride = torch.tensor(vals, device=next(m.parameters()).device)
                except Exception:
                    pass

    def to(self, device):
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()


class ChimeraDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo11-chimera.yaml", ch=3, nc=None, verbose=True, lambdas=None):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.lambdas = lambdas or {}
        # 기본 람다값
        self.lambdas.setdefault("nonmoving", 1.0)
        self.lambdas.setdefault("rider", 1.0)

        self._head_meta_built = False
        self.head_meta = {}  # head_name -> {global_ids, g2l, nc}


    def _build_head_meta_from_data(self):
        """
        self.data['multi_heads']를 읽어서
        head별로 global->local 클래스 매핑을 만든다.
        """

        if not hasattr(self, "_head_meta_built"):
            self._head_meta_built = False

        if self._head_meta_built:
            return

        if not hasattr(self, "data") or self.data is None:
            LOGGER.warning(
            "[ChimeraDetection] self.data 가 비어 있습니다. "
            "trainer/validator에서 model.data 를 세팅하기 전의 더미 predict 호출일 수 있습니다."
            )
            return

        mh_cfg = self.data.get("multi_heads", None)
        if mh_cfg is None:
            LOGGER.warning(
                "[ChimeraDetection] data.yaml 에 multi_heads 설정이 없습니다. "
                "ChimeraDetectionModel 을 사용하려면 data.yaml 에 multi_heads 를 정의해야 합니다."
            )
            return

        meta = {}
        for head_name, cfg in mh_cfg.items():
            class_ids = cfg.get("class_ids", None)
            if not class_ids:
                raise ValueError(f"multi_heads.{head_name}.class_ids 가 비어있습니다.")
            # global id -> head local id
            g2l = {int(g): i for i, g in enumerate(class_ids)}
            meta[head_name] = {
                "global_ids": [int(g) for g in class_ids],
                "g2l": g2l,
                "nc": len(class_ids),
            }

            # 람다 기본값 없으면 1.0
            self.lambdas.setdefault(head_name, 1.0)

        self.head_meta = meta
        self._head_meta_built = True

        if not hasattr(self, "_head_meta_debugged"):
            debug_meta = {
                k: {
                    "nc": v["nc"],
                    "global_ids": v["global_ids"],
                }
                for k, v in meta.items()
            }
            print("[ChimeraDetection][DEBUG] head_meta:", debug_meta)
            self._head_meta_debugged = True

        print("[DEBUG] type(self.data) =", type(self.data))
        print("[DEBUG] data.keys() =", list(self.data.keys()))
        print("[DEBUG] names =", self.data.get("names"))
        print("[DEBUG] type(names) =", type(self.data.get("names")))

    def _normalize_head_output(self, head_name: str, head_pred_raw):
        """
        head별 출력을 (B, A, C) 텐서로 정규화해서 돌려준다.
        - box_ch: 4
        - nc_head: 이 head가 담당하는 로컬 클래스 개수
        """
        if not hasattr(self, "data") or self.data is None or "multi_heads" not in getattr(self, "data", {}):
            if isinstance(head_pred_raw, tuple):
                decoded_like, raw_feats = head_pred_raw
                # decoded_like가 이미 (B, C, A)이면 그걸 그대로 쓰고, 아니면 detect._inference 사용
                if isinstance(decoded_like, torch.Tensor) and decoded_like.ndim == 3:
                    y = decoded_like
                else:
                    detect = self._head_to_detect.get(head_name, None) if hasattr(self, "_head_to_detect") else None
                    if detect is not None and isinstance(raw_feats, (list, tuple)):
                        y = detect._inference(list(raw_feats))
                    else:
                        raise RuntimeError("Chimera: dummy forward에서 head_pred_raw 해석 불가")
            elif isinstance(head_pred_raw, torch.Tensor) and head_pred_raw.ndim == 3:
                y = head_pred_raw
            else:
                # 최소한 크기만 맞는 텐서로 반환
                raise RuntimeError("Chimera: dummy forward fallback 처리 필요")
                
        # 1) head_meta 보장
        self._build_head_meta_from_data()
        if not hasattr(self, "_head_to_detect") or self._head_to_detect is None:
            self.init_criterion()

        # 2) head_pred_raw 해석
        if isinstance(head_pred_raw, tuple):
            # tuple 구조: (decoded_like, raw_feats)
            decoded_like, raw_feats = head_pred_raw

            # 여기서는 이미 디코딩된 출력(decoded_like)을 그대로 사용
            if not isinstance(decoded_like, torch.Tensor) or decoded_like.ndim != 3:
                raise TypeError(
                    f"[ChimeraDetection] head '{head_name}' decoded_like shape/type 이상: "
                    f"type={type(decoded_like)}, ndim={getattr(decoded_like, 'ndim', None)}"
                )
            y = decoded_like

            print(f"\n[TUPLE DEBUG] head={head_name}")
            print("  decoded_like.shape =", decoded_like.shape)
            if isinstance(raw_feats, (list, tuple)) and len(raw_feats) > 0:
                print("  raw_feats[0].shape =", raw_feats[0].shape)

        elif isinstance(head_pred_raw, (list, tuple)):
            # 이 경우에만 feature map list → _inference
            detect = self._head_to_detect[head_name]
            y = detect._inference(list(head_pred_raw))

        elif isinstance(head_pred_raw, torch.Tensor):
            # 이미 (B, C_tot, A) 텐서라면 그대로 사용
            if head_pred_raw.ndim != 3:
                raise ValueError(
                    f"[ChimeraDetection] head '{head_name}' tensor output ndim={head_pred_raw.ndim}, expected 3."
                )
            y = head_pred_raw

        else:
            raise TypeError(
                f"[ChimeraDetection] head '{head_name}' output type not supported: {type(head_pred_raw)}"
            )

        if not torch.isfinite(y).all():
            # 요약/더미 forward 같은 경우 여기서 정리
            y = torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)

        # 3) 이제 y는 (B, C_tot, A) 텐서여야 한다
        if not isinstance(y, torch.Tensor) or y.ndim != 3:
            raise ValueError(
                f"[ChimeraDetection] head '{head_name}' normalized y.ndim={getattr(y, 'ndim', None)}, expected 3."
            )

        B, C_tot, A = y.shape
        box_ch = 4
        nc_head = C_tot - box_ch
        if nc_head <= 0:
            raise ValueError(
                f"[ChimeraDetection] head '{head_name}' invalid C_tot={C_tot}, box_ch={box_ch}"
            )

        # 디버그
        print(f"\n[Norm DEBUG] head={head_name}")
        print("  y.shape =", y.shape)
        print("  box_ch =", box_ch, "nc_head =", nc_head)
        with torch.no_grad():
            box_all = y[:, :box_ch, :].flatten()  # (B*4*A,)
            print("  box_all_abs_sum =", float(box_all.abs().sum()))
            print("  box_all_max =", float(box_all.max()))
            print("  box_all_min =", float(box_all.min()))

            print("box_all_abs_sum_fp32 =", float(box_all.float().abs().sum()))
            print("has_inf =", bool(torch.isinf(box_all).any()))
            print("has_nan =", bool(torch.isnan(box_all).any()))

        # (B, C, A) → (B, A, C)
        y = y.permute(0, 2, 1).contiguous()
        return y, nc_head, box_ch


    def predict(self, x, augment=False, profile=False, visualize=False, embed=None):
        """
        멀티헤드 예측을 global class 53개로 합쳐서
        validator/NMS가 기대하는 단일 head 예측((B, A, 4+53))으로 반환.
        """
        # 0) 아직 data/multi_heads가 안 붙은 "cold start" 호출이면
        #    그냥 기본 DetectionModel.predict로 넘긴다 (stride warmup 등)
        if (
            not hasattr(self, "data")
            or self.data is None
            or "multi_heads" not in self.data
        ):
            return super().predict(x, augment=augment,
                                profile=profile,
                                visualize=visualize,
                                embed=embed)

        # 1) raw multi-head preds 얻기
        preds = self._predict_once(x, profile=profile, visualize=visualize, embed=embed)

        # 2) head_meta 준비 (여기서는 이미 self.data/multi_heads가 있다고 가정)
        self._build_head_meta_from_data()
        if not getattr(self, "head_meta", None):
            LOGGER.warning(
                "[ChimeraDetection] head_meta가 비어 있습니다. "
                "data.yaml의 multi_heads 설정을 확인하세요. "
                "임시로 기본 DetectionModel.predict를 사용합니다."
            )
            return super().predict(x, augment=augment,
                                profile=profile,
                                visualize=visualize,
                                embed=embed)

        if not hasattr(self, "_head_to_detect") or self._head_to_detect is None:
            self.init_criterion()

        # head 이름들 (예: ["nonmoving", "rider"])
        head_names = list(self.head_meta.keys())
        num_global_classes = len(self.data["names"])

        # preds 타입에 따라 dict로 정규화
        if isinstance(preds, dict):
            head_preds = preds
        else:
            head_preds = {name: p for name, p in zip(head_names, preds)}

        # 2) 기준 head = 첫 head
        ref_head = head_names[0]
        ref_flat, ref_nc, box_ch = self._normalize_head_output(ref_head, head_preds[ref_head])

        print("\n=== DEBUG C1: reference head flat ===")
        print("  ref_head =", ref_head)
        print("  flat.shape =", ref_flat.shape)  # (B, A, C)
        print("  ref_nc =", ref_nc)
        print("  box_ch =", box_ch)
        print("  expected total C =", box_ch + ref_nc)

        B, A, _ = ref_flat.shape

        # 3) global 출력 텐서 준비: (B, A, box_ch + num_global_classes)
        C_global = box_ch + num_global_classes
        global_flat = ref_flat.new_zeros((B, A, C_global))

        # 박스 부분은 기준 head의 것을 사용
        global_flat[..., :box_ch] = ref_flat[..., :box_ch]

        # 4) 각 head별 cls 로짓을 global class 인덱스로 매핑해서 채우기
        for head_name in head_names:
            flat, nc_head, box_ch_h = self._normalize_head_output(head_name, head_preds[head_name])
            assert box_ch_h == box_ch, f"box_ch mismatch: ref={box_ch}, {head_name}={box_ch_h}"

            meta = self.head_meta[head_name]
            global_ids = meta["global_ids"]  # 길이 = nc_head

            # local cls 채널: flat[..., box_ch : box_ch + nc_head]
            local_logits = flat[..., box_ch : box_ch + nc_head]  # (B, A, nc_head)

            # global_flat의 해당 위치에 그대로 삽입
            for local_idx, global_id in enumerate(global_ids):
                global_flat[..., box_ch + global_id] = local_logits[..., local_idx]

        # 5) 디버그: cls 통계
        with torch.no_grad():
            cls_logits = global_flat[..., box_ch:]  # (B, A, num_global)
            cls_logits_std = float(cls_logits.std())
            cls_min = float(cls_logits.min())
            cls_max = float(cls_logits.max())
            print("[ChimeraPredict DEBUG]  max_conf=", float(cls_logits.sigmoid().max()),
                "mean_conf=", float(cls_logits.sigmoid().mean()))
            print("[ChimeraPredict DEBUG2] cls_logits_std=", cls_logits_std,
                "cls_logits_min=", cls_min, "cls_logits_max=", cls_max)

        with torch.no_grad():
            boxes_debug = global_flat[0, :5, :4]  # (5, 4)
            print("\n[GLOBAL FLAT DEBUG] first5 boxes (xywh from ref_head):")
            print(boxes_debug)
            print("  nonzero_ratio =", float((boxes_debug.abs() > 1e-3).any(dim=-1).float().mean()))

        return global_flat


    def init_criterion(self):
        if hasattr(self, "det_criteria") and self.det_criteria:
            return self.det_criteria

        self._build_head_meta_from_data()
        head_names = list(self.head_meta.keys())  # 예: ["nonmoving", "rider"]

        sub_heads: list[tuple[str, Detect]] = []
        chimera_head = self.model[-1]

        for name, module in chimera_head.named_modules():
            # 자기 자신은 제외
            if module is chimera_head:
                continue
            if isinstance(module, Detect):
                sub_heads.append((name, module))

        print("[ChimeraDetection][DEBUG] Found Detect sub-heads:", [name for name, _ in sub_heads])

        if len(head_names) != len(sub_heads):
            raise ValueError(
                f"multi_heads({len(head_names)}) 와 Detect sub-head 수({len(sub_heads)}) 가 다릅니다.\n"
                f"multi_heads: {head_names}\n"
                f"sub_heads: {[name for name, _ in sub_heads]}"
            )

        self._head_to_detect = {}
        for head_name, (sub_name, detect_module) in zip(head_names, sub_heads):
            print(f"[ChimeraDetection][DEBUG] Map head '{head_name}' -> sub-head '{sub_name}'")
            self._head_to_detect[head_name] = detect_module
            # 🔥 여기서 stride 직접 복구
            # 모델 전체 stride를 가져올 수 있으면 그걸 우선 사용
            model_stride = getattr(self, "stride", None)  # 보통 tensor([8.,16.,32.]) 형태

            for head_name, (sub_name, detect_module) in zip(head_names, sub_heads):
                if hasattr(detect_module, "stride"):
                    s = detect_module.stride
                    # 텐서인 경우
                    if isinstance(s, torch.Tensor):
                        if (s == 0).all():
                            if isinstance(model_stride, torch.Tensor) and model_stride.numel() == s.numel():
                                detect_module.stride = model_stride.to(s.device)
                            else:
                                # fallback: 기본 [8, 16, 32]
                                nl = getattr(detect_module, "nl", 3)
                                base = 8.0
                                vals = [base * (2 ** i) for i in range(nl)]
                                detect_module.stride = torch.tensor(vals, device=s.device)
                    # 리스트/튜플인 경우
                    elif isinstance(s, (list, tuple)):
                        try:
                            if all(v == 0 for v in s):
                                if isinstance(model_stride, torch.Tensor) and model_stride.numel() == len(s):
                                    detect_module.stride = model_stride.to(next(detect_module.parameters()).device)
                                else:
                                    nl = getattr(detect_module, "nl", len(s))
                                    base = 8.0
                                    vals = [base * (2 ** i) for i in range(nl)]
                                    detect_module.stride = torch.tensor(vals, device=next(detect_module.parameters()).device)
                        except Exception:
                            pass

            # 디버그: 실제로 고쳐졌는지 다시 찍어보기 (한 번만 찍고 싶으면 flag 써도 됨)
            for head_name, (sub_name, detect_module) in zip(head_names, sub_heads):
                print(
                    f"[ChimeraDetection][DEBUG] (fixed) Detect config for head '{head_name}': "
                    f"sub_name={sub_name}, nc={getattr(detect_module, 'nc', None)}, "
                    f"stride={getattr(detect_module, 'stride', None)}, "
                    f"reg_max={getattr(detect_module, 'reg_max', None)}"
                )
        self.det_criteria = {}

        for head_name, (sub_name, detect_module) in zip(head_names, sub_heads):
            stride = getattr(detect_module, "stride", None)
            reg_max = getattr(detect_module, "reg_max", None)
            nc = getattr(detect_module, "nc", None)
            print(
                f"[ChimeraDetection][DEBUG] Detect config for head '{head_name}': "
                f"sub_name={sub_name}, nc={nc}, stride={stride}, reg_max={reg_max}"
            )
            proxy = _HeadLossProxy(detect_module, self.args)
            criterion = v8DetectionLoss(proxy)   # detect.nc/stride/reg_max 로 초기화됨
            self.det_criteria[head_name] = criterion
            self._head_to_detect[head_name] = detect_module
        print("[ChimeraDetection][DEBUG] criterion initialized for heads:", head_names)

        return self.det_criteria

    def _build_head_batch(self, full_batch: dict, head_name: str) -> dict:
        """
        full_batch에서 특정 head(nonmoving/rider)용 서브 배치를 만들어
        v8DetectionLoss에 그대로 넣을 수 있는 형태로 변환.

        full_batch[key] 구조:
            {
              "cls": (N_i, 1),
              "bboxes": (N_i, 4),
              "batch_idx": (N_i,),
              ...
            }
        """
        self._build_head_meta_from_data()
        if head_name not in self.head_meta:
            raise KeyError(f"Unknown head_name: {head_name}")

        meta = self.head_meta[head_name]
        global_ids = meta["global_ids"]
        g2l = meta["g2l"]

        # 원본 batch에서 공통 필드는 그대로 shallow copy
        head_batch = {k: v for k, v in full_batch.items()
                      if k not in ("cls", "bboxes", "batch_idx")}

        cls = full_batch["cls"]        # (N, 1)
        bboxes = full_batch["bboxes"]  # (N, 4)
        bidx = full_batch["batch_idx"] # (N,)

        if cls.numel() == 0:
            # 이 batch 자체에 아무 object가 없는 경우
            head_batch["cls"] = cls.new_zeros((0, 1))
            head_batch["bboxes"] = bboxes.new_zeros((0, 4))
            head_batch["batch_idx"] = bidx.new_zeros((0,))
            return head_batch

        # 1) global class 기준으로 이 head가 관심있는 것만 필터링
        cls_flat = cls.view(-1).to(torch.long)
        mask = torch.isin(cls_flat, torch.tensor(global_ids, device=cls.device))

        if not mask.any():
            # 이 head가 볼 object가 하나도 없는 case
            head_batch["cls"] = cls.new_zeros((0, 1))
            head_batch["bboxes"] = bboxes.new_zeros((0, 4))
            head_batch["batch_idx"] = bidx.new_zeros((0,))
            return head_batch

        # 2) 필터링된 타겟만 골라냄
        cls_filtered = cls_flat[mask]
        bboxes_filtered = bboxes[mask]
        bidx_filtered = bidx[mask]

        # 3) global id -> head local id로 매핑
        #    (0~nc_head-1)
        mapped = [g2l[int(c)] for c in cls_filtered.cpu().tolist()]
        cls_head = torch.tensor(mapped, device=cls.device, dtype=cls.dtype).view(-1, 1)

        head_batch["cls"] = cls_head
        head_batch["bboxes"] = bboxes_filtered
        head_batch["batch_idx"] = bidx_filtered
        # 🔍 디버그: head별 타깃 통계 (각 head에 대해 한 번만)
        debug_flag_name = f"_debug_head_batch_{head_name}"
        if not hasattr(self, debug_flag_name):
            n = cls_head.shape[0]
            if n > 0:
                cls_min = int(cls_head.min().item())
                cls_max = int(cls_head.max().item())
                x_min = float(bboxes_filtered[:, 0].min().item())
                y_min = float(bboxes_filtered[:, 1].min().item())
                x_max = float(bboxes_filtered[:, 2].max().item())
                y_max = float(bboxes_filtered[:, 3].max().item())
            else:
                cls_min = cls_max = None
                x_min = y_min = x_max = y_max = None

            print(
                f"[ChimeraDetection][DEBUG] head_batch('{head_name}'): "
                f"N={n}, cls_range=[{cls_min}, {cls_max}], "
                f"bbox_x=[{x_min}, {x_max}], bbox_y=[{y_min}, {y_max}]"
            )
            setattr(self, debug_flag_name, True)
        return head_batch

    def loss(self, batch, preds=None):
        if not hasattr(self, "det_criteria_initialized"):
            self.init_criterion()
            self.det_criteria_initialized = True
        # 0) head 메타 / criterion 준비
        self._build_head_meta_from_data()

        img = batch["img"]

        # 1) 항상 멀티헤드 raw prediction을 새로 얻는다.
        multi_head_out = self._predict_once(img, profile=False, visualize=False, embed=None)

        
        head_names = list(self.head_meta.keys())  # ["nonmoving", "rider", ...]

        # 2) multi_head_out → head_name -> head_pred 로 정규화
        if isinstance(multi_head_out, dict):
            head_preds = multi_head_out
        elif isinstance(multi_head_out, (list, tuple)):
            if len(multi_head_out) < len(head_names):
                raise ValueError(
                    f"예측된 head 수({len(multi_head_out)})가 multi_heads 정의({len(head_names)})보다 작습니다."
                )
            head_preds = {name: p for name, p in zip(head_names, multi_head_out)}
        else:
            raise TypeError(f"예상치 못한 multi_head_out 타입: {type(multi_head_out)}")

        device = img.device
        total_vec = torch.zeros(3, device=device)  # [box, cls, dfl]

        for head_name in head_names:
            if head_name not in head_preds:
                continue

            head_pred = head_preds[head_name]               # 이게 [P3, P4, P5] 구조
            head_batch = self._build_head_batch(batch, head_name)
            if head_batch is None:
                continue

            crit = self.det_criteria[head_name]             # v8DetectionLoss(proxy)
            # v8DetectionLoss는 (loss_vec(3,), loss_items(3,))를 리턴
            loss_vec, _ = crit(head_pred, head_batch)

            lam = self.lambdas.get(head_name, 1.0)
            total_vec = total_vec + lam * loss_vec

        total = total_vec.sum()
        loss_items = total_vec.detach()
        return total, loss_items

    def postprocess(self, preds):
        """
        preds: ChimeraPredict 에서 넘어온 (B, A, 4+53) 텐서라고 가정하고,
        여기서 NMS + dict 포맷 변환을 전부 직접 수행한다.
        """

        # H1: 어떤 경로로 들어오는지 1회만 확인
        if not hasattr(self, "_chimera_post_debug_once"):
            self._chimera_post_debug_once = True
            print("\n=== DEBUG H1: postprocess input ===")
            print("  preds type =", type(preds))
            if isinstance(preds, torch.Tensor):
                print("  preds.shape =", preds.shape)
                B, A, C = preds.shape
                print("  B, A, C =", B, A, C)
                print("  first anchor cls logits (first 10):",
                      preds[0, 0, 4:14].tolist())
            else:
                print("  (unexpected preds type)")

        # 이미 list[dict]면, 이 postprocess 경로가 아니라 다른 경로에서 한 번 더 돌고 있다는 뜻
        if not isinstance(preds, torch.Tensor):
            print("[WARN] ChimeraDetection.postprocess got non-Tensor, bypassing.")
            return preds

        # ---------- 1) YOLO 표준 NMS ----------
        nms_out = ops.non_max_suppression(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            classes=self.args.classes,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

        if not hasattr(self, "_chimera_post_nms_once"):
            self._chimera_post_nms_once = True
            print("\n=== DEBUG H2: after NMS ===")
            print("  type(nms_out) =", type(nms_out))
            if len(nms_out) > 0 and isinstance(nms_out[0], torch.Tensor):
                print("  nms_out[0].shape =", nms_out[0].shape)
                if nms_out[0].numel() > 0:
                    print("  nms_out[0][0] =", nms_out[0][0].tolist())

        # ---------- 2) dict 포맷으로 포장 ----------
        packed = []
        nc = len(self.data["names"])  # 53

        for det in nms_out:
            if not isinstance(det, torch.Tensor) or det.numel() == 0:
                packed.append(
                    {
                        "bboxes": torch.zeros((0, 4), device=self.device),
                        "conf": torch.zeros((0,), device=self.device),
                        "cls": torch.zeros((0,), dtype=torch.long, device=self.device),
                    }
                )
                continue

            boxes = det[:, :4]
            conf = det[:, 4]
            cls  = det[:, 5].long()

            if not hasattr(self, "_chimera_post_cls_once"):
                self._chimera_post_cls_once = True
                print("\n=== DEBUG H3: cls index before clamp ===")
                print("  cls min/max =", int(cls.min()), int(cls.max()))
                print("  cls[:10] =", cls[:10].tolist())

            valid = (cls >= 0) & (cls < nc)
            boxes = boxes[valid]
            conf  = conf[valid]
            cls   = cls[valid]

            packed.append({"bboxes": boxes, "conf": conf, "cls": cls})

        if not hasattr(self, "_chimera_post_out_once"):
            self._chimera_post_out_once = True
            print("\n=== DEBUG H4: postprocess output ===")
            print("  type(packed) =", type(packed))
            if len(packed) > 0:
                p0 = packed[0]
                print("  item[0].keys =", p0.keys())
                c0 = p0["cls"]
                print("  item[0].cls[:10] =", c0[:10].tolist() if c0.numel() > 0 else [])
                print("  item[0].cls min/max =",
                      int(c0.min()) if c0.numel() > 0 else -1,
                      int(c0.max()) if c0.numel() > 0 else -1)

        return packed

class OBBModel(DetectionModel):
    """YOLO Oriented Bounding Box (OBB) model.

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized loss
    computation for rotated object detection.

    Methods:
        __init__: Initialize YOLO OBB model.
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model
        >>> model = OBBModel("yolo11n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO segmentation model.

    This class extends DetectionModel to handle instance segmentation tasks, providing specialized loss computation for
    pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLO segmentation model.
        init_criterion: Initialize the loss criterion for segmentation.

    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize Ultralytics YOLO segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLO pose model.

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized loss computation for
    keypoint detection and pose estimation.

    Attributes:
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).

    Methods:
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.

    Examples:
        Initialize a pose model
        >>> model = PoseModel("yolo11n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize Ultralytics YOLO Pose model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            data_kpt_shape (tuple): Shape of keypoints data.
            verbose (bool): Whether to display model information.
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks, providing model
    initialization, configuration, and output reshaping capabilities.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        stride (torch.Tensor): Model stride values.
        names (dict): Class names dictionary.

    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a classification model
        >>> model = ClassificationModel("yolo11n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):
        """Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set Ultralytics YOLO model configurations and define the model architecture.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required.

        Args:
            model (torch.nn.Module): Model to update.
            nc (int): New number of classes.
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.

    Methods:
        __init__: Initialize the RTDETRDetectionModel.
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.
        predict: Perform forward pass through the model.

    Examples:
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def _apply(self, fn):
        """Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (RTDETRDetectionModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]
        m.anchors = fn(m.anchors)
        m.valid_mask = fn(m.valid_mask)
        return self

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions.

        Returns:
            loss_sum (torch.Tensor): Total loss value.
            loss_items (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model.

    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based class
    specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.

    Examples:
        Initialize a world model
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts, supporting
    both prompt-based and prompt-free inference modes.

    Attributes:
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOE model.
        get_text_pe: Get text positional embeddings.
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.

    Examples:
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
            without_reprta (bool): Whether to return text embeddings cooperated with reprta module.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("mobileclip:blt", device=device)

        model = self.clip_model if cache_clip_model else build_text_model("mobileclip:blt", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # run auxiliary text head

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """Get visual embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.

        Returns:
            (torch.Tensor): Visual positional embeddings.
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """Set vocabulary for the prompt-free model.

        Args:
            vocab (nn.ModuleList): List of vocabulary items.
            names (list[str]): List of class names.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # Cache anchors for head
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

        # re-parameterization for prompt-free model
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2)
            for i, (cls, pf, loc) in enumerate(zip(vocab, head.cv3, head.cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """Get fused vocabulary layer from the model.

        Args:
            names (list): List of class names.

        Returns:
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

        vocab = nn.ModuleList()
        for cls_head in head.cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            names (list[str]): List of class names.
            embeddings (torch.Tensor): Embeddings tensor.
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """Get class positional embeddings.

        Args:
            tpe (torch.Tensor, optional): Text positional embeddings.
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x = m(x, cls_pe)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPDetectLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE segmentation model.

    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts, providing
    specialized loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLOE segmentation model.
        loss: Compute loss with prompts for segmentation.

    Examples:
        Initialize a YOLOE segmentation model
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPSegmentLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through model averaging
    or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            y (torch.Tensor): Concatenated predictions from all models.
            train_out (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code, where you've
    moved a module from one location to another, but you still want to support the old import paths for backwards
    compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Notes:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules.

        Args:
            module (str): Module name.
            name (str): Class name.

        Returns:
            (type): Found class or SafeClass.
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches
    the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """Load a single model weights.

    Args:
        weight (str | Path): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model

    # Model compatibility updates
    model.args = args  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect, ChimeraDetect}
        ):
            args.append([ch[x] for x in f])
            if m is Segment or m is YOLOESegment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ChimeraDetect}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x).
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""


def guess_model_task(model):
    """Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect

