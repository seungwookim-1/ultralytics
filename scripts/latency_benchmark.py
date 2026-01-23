import time
import torch
import numpy as np
from ultralytics import YOLO

def benchmark_latency(model: YOLO, imgsz=640, runs=100):
    # 1) device 결정: 첫 번째 파라미터의 device 사용
    base_module = model.model  # DetectionModel
    first_param = next(base_module.parameters())
    device = first_param.device

    base_module.eval()

    # 2) dummy 입력 생성 (B=1)
    dummy = torch.rand(1, 3, imgsz, imgsz, device=device)

    # AMP/half precision이면 맞춰줌
    if first_param.dtype == torch.float16:
        dummy = dummy.half()

    # 3) warmup
    with torch.no_grad():
        for _ in range(10):
            _ = base_module(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 4) 측정
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = base_module(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    ms_per_img = (time.perf_counter() - t0) / runs * 1000
    return ms_per_img

if __name__ == "__main__":
    model_pairs = [
        ("YOLOn", "/ultralytics/outputs/scheduling_top_k_2_test_1/results/YOLOn_multi_s11/weights/best.pt"),
        ("MoE", "/ultralytics/outputs/scheduling_top_k_2_test_1/results/MoE_multi_BAL2.00_ENT0.08_AUX0.030_NS0.015_SCHED_lambda_balance_min0.85_max1.30_s11/weights/best.pt"),
        # ("YOLOs", "/ultralytics/runs/multi_11n_11s_moe/YOLOs_multi_s0/weights/best.pt"),
    ]
    for model, weight in model_pairs:
        print(f"{model} latency: {benchmark_latency(YOLO(weight))} ms/img")
        
