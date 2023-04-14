from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Yolov8 DOCS https://docs.ultralytics.com/modes/train/
if __name__ == "__main__":
    # model = YOLO("yolov8m.pt")
    # model.train(data="nematodes.yaml", epochs=100, imgsz=640)

    model_large = YOLO("yolov8n.pt")
    model_large.train(data="our_dataset.yaml", epochs=100, imgsz=640, batch=48, name= "our_tiny")

    model_large = YOLO("yolov8m.pt")
    model_large.train(data="our_dataset.yaml", epochs=100, imgsz=640, batch=16, name= "our_medium")

    model_large = YOLO("yolov8x.pt")
    model_large.train(data="our_dataset.yaml", epochs=100, imgsz=640, batch=2, name= "our_extra")