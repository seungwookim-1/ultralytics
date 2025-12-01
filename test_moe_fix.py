#!/usr/bin/env python3
"""Test that MoEDetect forward pass works correctly in all modes."""

import torch
from ultralytics.nn.modules.head import MoEDetect

print("=" * 70)
print("Testing MoEDetect Forward Pass Fix")
print("=" * 70)

ch = (256, 512, 1024)
moe_head = MoEDetect(nc=80, ch=ch, num_experts=4)
moe_head.stride = torch.tensor([8, 16, 32])

x = [
    torch.randn(2, 256, 80, 80),
    torch.randn(2, 512, 40, 40),
    torch.randn(2, 1024, 20, 20),
]

# Test 1: Building stage (stride == 0)
print("\n[Test 1] Building Stage (stride == 0)")
moe_head.stride = torch.zeros(3)
result = moe_head(x)
assert isinstance(result, list), "Building stage should return list"
assert len(result) == 3, "Should have 3 outputs"
print("  ✓ Building stage returns list of outputs")
print(f"  ✓ Output shapes: {[r.shape for r in result]}")

# Reset stride for other tests
moe_head.stride = torch.tensor([8, 16, 32])

# Test 2: Training mode
print("\n[Test 2] Training Mode")
moe_head.train()
result = moe_head(x)
assert isinstance(result, tuple), "Training should return tuple"
assert len(result) == 2, "Should return (outputs, aux_loss)"
outputs, aux_loss = result
assert len(outputs) == 3, "Should have 3 scale outputs"
assert isinstance(aux_loss, torch.Tensor), "Aux loss should be tensor"
print("  ✓ Training mode returns (outputs, aux_loss)")
print(f"  ✓ Aux loss: {aux_loss.item():.6f}")

# Test 3: Validation mode (not training, not export)
print("\n[Test 3] Validation/Inference Mode")
moe_head.eval()
moe_head.export = False
with torch.no_grad():
    result = moe_head(x)

assert isinstance(result, tuple), "Validation should return tuple"
assert len(result) == 2, "Should return (y, outputs)"
y, raw_outputs = result
print("  ✓ Validation mode returns (y, outputs)")
print(f"  ✓ Inference output (y) shape: {y.shape}")
print(f"  ✓ Raw outputs count: {len(raw_outputs)}")

# Verify y is properly formatted for NMS
assert y.dim() == 3, "Inference output should be 3D [B, N, C]"
print(f"  ✓ Inference output is properly formatted: {y.shape}")
print(f"  ✓ Format: [batch={y.shape[0]}, detections={y.shape[1]}, data={y.shape[2]}]")

# Test 4: Export mode
print("\n[Test 4] Export Mode")
moe_head.export = True
with torch.no_grad():
    result = moe_head(x)

assert isinstance(result, torch.Tensor), "Export should return tensor"
assert result.dim() == 3, "Export output should be 3D"
print("  ✓ Export mode returns inference tensor")
print(f"  ✓ Export output shape: {result.shape}")

print("\n" + "=" * 70)
print("✅ All forward pass tests passed!")
print("=" * 70)
print("\nThe fix is working correctly:")
print("  - Building stage: Returns raw list")
print("  - Training: Returns (outputs, aux_loss)")
print("  - Validation: Returns (y, outputs) - FIXED for NMS")
print("  - Export: Returns y only")
print("\nThe validation mode now returns inference-formatted output")
print("that is compatible with NMS post-processing.")
print("=" * 70)
