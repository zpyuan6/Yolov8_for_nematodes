from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data="nematodes.yaml", epochs=100, imgsz=640)
