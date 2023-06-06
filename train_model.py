from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Yolov8 DOCS https://docs.ultralytics.com/modes/train/
if __name__ == "__main__":
    # model = YOLO("yolov8m.pt")
    # model.train(data="nematodes.yaml", epochs=100, imgsz=640)
    # model_list = ["yolov8n.pt","yolov8m.pt","yolov8x.pt"]
    model_list = ["runs\\detect\\our_nemo_tiny_all2\\weights\\best.pt","runs\\detect\\our_nemo_medium_all\\weights\\best.pt","runs\\detect\\our_nemo_extra_all\\weights\\best.pt"]

    model_large = YOLO(model_list[0])
    model_large.train(data="our_nematodes.yaml", epochs=200, imgsz=512, batch=48, name= "our_nemo_tiny_all_05_06")

    # model_large = YOLO(model_list[1])
    # model_large.train(data="our_nematodes.yaml", epochs=100, imgsz=640, batch=16, name= "our_nemo_medium_all_05_06")

    # model_large = YOLO(model_list[2])
    # model_large.train(data="our_nematodes.yaml", epochs=50, imgsz=640, batch=2, name= "our_nemo_extra_all_05_06")