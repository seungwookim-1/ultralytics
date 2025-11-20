import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple
import json
from ultralytics.models.yolo.detect.chimera import ChimeraYOLO
# Import your multi-head model
# from multihead_yolo import ChimeraYOLO


class CityscapesMultiHeadTester:
    """
    Test multi-head YOLO model on Cityscapes dataset.
    """
    
    # Cityscapes-relevant class mappings for multi-head detection
    HEAD_CONFIGS = {
        "vehicle": {
            "nc": 8,
            "classes": ["car", "truck", "bus", "train", "motorcycle", "bicycle", "caravan", "trailer"]
        },
        "vru": {  # Vulnerable Road Users
            "nc": 2,
            "classes": ["person", "rider"]
        },
        "traffic": {
            "nc": 4,
            "classes": ["traffic_light", "traffic_sign", "pole", "polegroup"]
        }
    }
    
    COLORS = {
        "vehicle": (0, 255, 0),    # Green
        "vru": (255, 0, 0),        # Red
        "traffic": (0, 0, 255)     # Blue
    }
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        """
        Initialize tester.
        
        Args:
            model_path: Path to trained model weights
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str = None):
        """Load or create multi-head model."""
        from ultralytics.models.yolo.detect.chimera import ChimeraYOLO
        
        # Create model
        head_defs = {
            name: {"type": "detect", "nc": cfg["nc"]} 
            for name, cfg in self.HEAD_CONFIGS.items()
        }
        
        lambdas = {
            "vehicle": 1.0,
            "vru": 1.5,
            "traffic": 2.0
        }
        
        model = ChimeraYOLO(
            cfg='yolo11n.yaml',
            head_defs=head_defs,
            lambdas=lambdas
        )
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            print(f"Loading weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Using randomly initialized model (for testing pipeline only)")
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, img_path: str, img_size: int = 640) -> Tuple[torch.Tensor, np.ndarray, Tuple]:
        """
        Preprocess Cityscapes image for YOLO.
        
        Args:
            img_path: Path to image
            img_size: Target size for model input
            
        Returns:
            Preprocessed tensor, original image, (original_h, original_w)
        """
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_shape = img.shape[:2]
        
        # Resize with padding to maintain aspect ratio
        h, w = img.shape[:2]
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Pad to square
        pad_h = img_size - new_h
        pad_w = img_size - new_w
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, img, orig_shape
    
    def postprocess_predictions(self, outputs: Dict, orig_shape: Tuple, img_size: int = 640, 
                                conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> Dict:
        """
        Postprocess multi-head predictions.
        
        Args:
            outputs: Dict of predictions from each head
            orig_shape: Original image shape (h, w)
            img_size: Model input size
            conf_thresh: Confidence threshold
            iou_thresh: IoU threshold for NMS
            
        Returns:
            Dict of processed detections per head
        """
        detections = {}
        
        for head_name, pred in outputs.items():
            # pred shape: [batch, num_boxes, 4 + nc]
            # 4 bbox coords + nc class scores
            
            if isinstance(pred, (list, tuple)):
                pred = pred[0]  # Take first element if it's inference output
            
            # Filter by confidence
            if pred.shape[-1] > 4:
                boxes = pred[0, :, :4]  # [num_boxes, 4] (x, y, w, h)
                scores = pred[0, :, 4:].max(dim=1)  # [num_boxes]
                class_ids = pred[0, :, 4:].argmax(dim=1)
                
                # Filter by confidence
                mask = scores.values > conf_thresh
                boxes = boxes[mask]
                scores = scores.values[mask]
                class_ids = class_ids[mask]
                
                # Apply NMS
                if len(boxes) > 0:
                    keep_indices = self._nms(boxes, scores, iou_thresh)
                    boxes = boxes[keep_indices]
                    scores = scores[keep_indices]
                    class_ids = class_ids[keep_indices]
                    
                    # Scale boxes back to original image size
                    boxes = self._scale_boxes(boxes, img_size, orig_shape)
                    
                    detections[head_name] = {
                        'boxes': boxes.cpu().numpy(),
                        'scores': scores.cpu().numpy(),
                        'class_ids': class_ids.cpu().numpy()
                    }
                else:
                    detections[head_name] = {
                        'boxes': np.array([]),
                        'scores': np.array([]),
                        'class_ids': np.array([])
                    }
            else:
                detections[head_name] = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'class_ids': np.array([])
                }
        
        return detections
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
        """Non-Maximum Suppression."""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0]
            keep.append(i.item())
            
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            
            w = torch.maximum(torch.tensor(0.0, device=boxes.device), xx2 - xx1)
            h = torch.maximum(torch.tensor(0.0, device=boxes.device), yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            indices = torch.where(iou <= iou_thresh)[0]
            order = order[indices + 1]
        
        return torch.tensor(keep, device=boxes.device)
    
    def _scale_boxes(self, boxes: torch.Tensor, img_size: int, orig_shape: Tuple) -> torch.Tensor:
        """Scale boxes from model input size to original image size."""
        h, w = orig_shape
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        pad_h = img_size - new_h
        pad_w = img_size - new_w
        top = pad_h // 2
        left = pad_w // 2
        
        # Convert from center format to corner format
        boxes[:, 0] = (boxes[:, 0] - left) / scale  # x center
        boxes[:, 1] = (boxes[:, 1] - top) / scale   # y center
        boxes[:, 2] = boxes[:, 2] / scale           # width
        boxes[:, 3] = boxes[:, 3] / scale           # height
        
        return boxes
    
    def visualize_detections(self, img: np.ndarray, detections: Dict, 
                           save_path: str = None, show: bool = True):
        """
        Visualize multi-head detections.
        
        Args:
            img: Original image
            detections: Dict of detections per head
            save_path: Path to save visualization
            show: Whether to display the image
        """
        fig, ax = plt.subplots(1, figsize=(15, 10))
        ax.imshow(img)
        
        for head_name, dets in detections.items():
            if len(dets['boxes']) == 0:
                continue
            
            color = self.COLORS.get(head_name, (255, 255, 0))
            color_normalized = tuple(c / 255.0 for c in color)
            
            for box, score, class_id in zip(dets['boxes'], dets['scores'], dets['class_ids']):
                x_center, y_center, w, h = box
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2, edgecolor=color_normalized,
                    facecolor='none', linestyle='-'
                )
                ax.add_patch(rect)
                
                # Add label
                class_name = self.HEAD_CONFIGS[head_name]['classes'][int(class_id)]
                label = f"{head_name}:{class_name} {score:.2f}"
                ax.text(
                    x1, y1 - 5, label,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color_normalized, alpha=0.7),
                    fontsize=8, color='white', weight='bold'
                )
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def test_single_image(self, img_path: str, save_dir: str = None, 
                         conf_thresh: float = 0.25) -> Dict:
        """
        Test on a single Cityscapes image.
        
        Args:
            img_path: Path to image
            save_dir: Directory to save results
            conf_thresh: Confidence threshold
            
        Returns:
            Dictionary with detection results
        """
        print(f"\nProcessing: {img_path}")
        
        # Preprocess
        img_tensor, orig_img, orig_shape = self.preprocess_image(img_path)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Postprocess
        detections = self.postprocess_predictions(
            outputs, orig_shape, conf_thresh=conf_thresh
        )
        
        # Print results
        print("\nDetection Results:")
        for head_name, dets in detections.items():
            num_dets = len(dets['boxes'])
            print(f"  {head_name}: {num_dets} detections")
            if num_dets > 0:
                for i, (score, class_id) in enumerate(zip(dets['scores'], dets['class_ids'])):
                    class_name = self.HEAD_CONFIGS[head_name]['classes'][int(class_id)]
                    print(f"    - {class_name}: {score:.3f}")
        
        # Visualize
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            img_name = Path(img_path).stem
            save_path = save_dir / f"{img_name}_result.jpg"
            self.visualize_detections(orig_img, detections, save_path=str(save_path), show=False)
        else:
            self.visualize_detections(orig_img, detections, show=True)
        
        return detections
    
    def test_directory(self, img_dir: str, save_dir: str = None, 
                      max_images: int = None, conf_thresh: float = 0.25):
        """
        Test on directory of Cityscapes images.
        
        Args:
            img_dir: Directory containing images
            save_dir: Directory to save results
            max_images: Maximum number of images to process
            conf_thresh: Confidence threshold
        """
        img_dir = Path(img_dir)
        image_paths = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Found {len(image_paths)} images in {img_dir}")
        
        all_results = {}
        for img_path in image_paths:
            try:
                detections = self.test_single_image(
                    str(img_path), 
                    save_dir=save_dir, 
                    conf_thresh=conf_thresh
                )
                all_results[img_path.name] = detections
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        # Save summary
        if save_dir:
            summary_path = Path(save_dir) / "detection_summary.json"
            summary = {
                img_name: {
                    head: {
                        'num_detections': len(dets['boxes']),
                        'classes': [
                            self.HEAD_CONFIGS[head]['classes'][int(cid)] 
                            for cid in dets['class_ids']
                        ]
                    }
                    for head, dets in detections.items()
                }
                for img_name, detections in all_results.items()
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSaved summary to {summary_path}")
        
        return all_results


# Example usage
if __name__ == "__main__":
    # Define heads
    head_defs = {
        "vehicle": {"type": "detect", "nc": 8},
        "vru": {"type": "detect", "nc": 2},
        "traffic_sign": {"type": "detect", "nc": 12}
    }
    
    # Define loss weights
    lambdas = {
        "vehicle": 1.0,
        "vru": 1.5,
        "traffic_sign": 2.0
    }
    
    # Create model
    model = ChimeraYOLO(
        cfg='yolo11n.yaml',
        head_defs=head_defs,
        lambdas=lambdas,
        verbose=True
    )
    
    # Dummy input
    x = torch.randn(2, 3, 640, 640)
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        print("\nInference outputs:")
        for head_name, pred in outputs.items():
            if isinstance(pred, (list, tuple)):
                print(f"  {head_name}: list of {len(pred)} tensors")
                for i, p in enumerate(pred):
                    print(f"    [{i}]: {p.shape}")
            else:
                print(f"  {head_name}: {pred.shape}")
    
    # Training mode
    model.train()
    dummy_targets = {
        "vehicle": torch.randn(2, 50, 6),
        "vru": torch.randn(2, 20, 6),
        "traffic_sign": torch.randn(2, 30, 6)
    }
    
    loss_dict = model(x, dummy_targets)
    print(f"\nTraining loss: {loss_dict['total_loss'].item():.4f}")
    print("Per-head losses:")
    for head_name, loss in loss_dict['per_head'].items():
        print(f"  {head_name}: {loss.item():.4f}")