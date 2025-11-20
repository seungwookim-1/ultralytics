import torch
from ultralytics.models.yolo.detect.chimera import ChimeraYOLO

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
        print("Inference outputs:")
        for head_name, pred in outputs.items():
            print(f"  {head_name}: {pred.shape if isinstance(pred, torch.Tensor) else 'tuple'}")
    
    # Training mode
    model.train()
    dummy_targets = {
        "vehicle": torch.randn(2, 50, 6),  # Dummy targets
        "vru": torch.randn(2, 20, 6),
        "traffic_sign": torch.randn(2, 30, 6)
    }
    
    loss_dict = model(x, dummy_targets)
    print(f"\nTraining loss: {loss_dict['total_loss'].item():.4f}")
    print("Per-head losses:")
    for head_name, loss in loss_dict['per_head'].items():
        print(f"  {head_name}: {loss.item():.4f}")
