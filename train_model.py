from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    model.train(data="nematodes.yaml", epochs=100, imgsz=640)

    model_large = YOLO("yolov8x.pt")
    model_large.train(data="nematodes.yaml", epochs=100, imgsz=640)