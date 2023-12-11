from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch

# Yolov8 DOCS https://docs.ultralytics.com/modes/train/
if __name__ == "__main__":
    # model = YOLO("yolov8m.pt")
    # model.train(data="nematodes.yaml", epochs=100, imgsz=640)
    model_list = ["yolov8n.pt","yolov8m.pt","yolov8x.pt"]
    # model_list = ["/home/zhipeng/Desktop/nametodes/Yolov8_for_nematodes/runs/detect/pest_uk_tiny_07_06/weights/best.pt","/home/zhipeng/Desktop/nametodes/Yolov8_for_nematodes/runs/detect/pest_uk_medium_07_06/weights/best.pt","/home/zhipeng/Desktop/nametodes/Yolov8_for_nematodes/runs/detect/pest_uk_extra_07_06/weights/best.pt"]

    model_large = YOLO("models_config/yolov8n_cbam_attention.yaml")
    weight_model = torch.load(model_list[0])
    print(type(weight_model['model']))
    
    model_large.model.load(weight_model['model'])
    model_large.train(data="nematodes.yaml", epochs=400, imgsz=640, batch=32, name= "nematodes_tiny_attention_11_Dec")

    # model_large = YOLO(model_list[1])
    # model_large.train(data="nematodes.yaml", epochs=200, imgsz=640, batch=16, name= "nematodes_medium_31_08")

    # model_large = YOLO(model_list[2])
    # model_large.train(data="nematodes.yaml", epochs=200, imgsz=640, batch=2, name= "nematodes_extra_31_08")
    # model_list = ["runs\\detect\\our_nemo_tiny_all2\\weights\\best.pt","runs\\detect\\our_nemo_medium_all\\weights\\best.pt","runs\\detect\\our_nemo_extra_all\\weights\\best.pt"]

    # model_large = YOLO(model_list[0])
    # model_large.train(data="microorganism.yaml", epochs=400, imgsz=512, batch=48, name= "microorganism_24_07")

    # model_large = YOLO(model_list[0])
    # model_large.train(data="BBBC010.yaml", epochs=400, imgsz=512, batch=48, name= "BBBC010_24_07")

    # model_large = YOLO(model_list[1])
    # model_large.train(data="our_nematodes.yaml", epochs=200, imgsz=640, batch=16, name= "our_nemo_medium_06_06")

    # model_large = YOLO(model_list[2])
    # model_large.train(data="our_nematodes.yaml", epochs=100, imgsz=640, batch=2, name= "our_nemo_extra_06_06")
