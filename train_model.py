from ultralytics import YOLO
import torch


if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data="nematodes.yaml", epochs=100, imgsz=640)
