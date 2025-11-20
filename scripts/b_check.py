from ultralytics import YOLO
mA = YOLO("runs/A_teacher/weights/best.pt").model
mB = YOLO("runs/B_with_distill/weights/best.pt").model

print("A nc:", mA.model[-1].nc, "names:", len(mA.names))
print("B nc:", mB.model[-1].nc, "names:", len(mB.names))