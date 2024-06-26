from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch

def training_original_yolo(dataset_yaml, folder_name):
    model_list = ["yolov8n.pt","yolov8m.pt","yolov8x.pt"]

    model_large = YOLO(model_list[0])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=16, name= f"{folder_name}_tiny")

    model_large = YOLO(model_list[1])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=16, name= f"{folder_name}_medium")

    model_large = YOLO(model_list[2])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=2, name= f"{folder_name}_extra")


# Yolov8 DOCS https://docs.ultralytics.com/modes/train/
if __name__ == "__main__":

    training_original_yolo("uk_pest_dataset_26MAR.yaml","uk_pest_26MAR")

    # model = YOLO("yolov8m.pt")
    # model.train(data="nematodes.yaml", epochs=100, imgsz=640)
    # model_list = ["yolov8n.pt","yolov8m.pt","yolov8x.pt"]
    # model_list = ["/home/zhipeng/Desktop/nametodes/Yolov8_for_nematodes/runs/detect/pest_uk_tiny_07_06/weights/best.pt","/home/zhipeng/Desktop/nametodes/Yolov8_for_nematodes/runs/detect/pest_uk_medium_07_06/weights/best.pt","/home/zhipeng/Desktop/nametodes/Yolov8_for_nematodes/runs/detect/pest_uk_extra_07_06/weights/best.pt"]

    # model_large = YOLO("models_config/yolov8n_cbam_attention.yaml")
    # weight_model = torch.load(model_list[0])
    # print(type(weight_model['model']))
    
    # model_large.model.load(weight_model['model'])
    # model_large.train(data="nematodes.yaml", epochs=400, imgsz=640, batch=32, name= "nematodes_tiny_attention_11_Dec")


    # model_list = ["runs\\detect\\our_nemo_tiny_all2\\weights\\best.pt","runs\\detect\\our_nemo_medium_all\\weights\\best.pt","runs\\detect\\our_nemo_extra_all\\weights\\best.pt"]

    # model_large = YOLO(model_list[0])
    # model_large.train(data="microorganism.yaml", epochs=400, imgsz=512, batch=48, name= "microorganism_24_07")

#     model_list = [
#             "/home/zhipeng/Desktop/nematodes/Yolov8_for_nematodes/runs/detect/uk_pest_24DEC_tiny/weights/best.pt",
#             "yolov8m.pt",
#             "yolov8x.pt",
#             "/home/zhipeng/Desktop/nematodes/Yolov8_for_nematodes/runs/detect/uk_pest_29DEC_tiny/weights/best.pt",
#             "/home/zhipeng/Desktop/nematodes/Yolov8_for_nematodes/runs/detect/uk_pest_29DEC_medium/weights/best.pt",
#             "/home/zhipeng/Desktop/nematodes/Yolov8_for_nematodes/runs/detect/uk_pest_29DEC_extra/weights/best.pt"
#             ]

#     model_large = YOLO(model_list[0])
#     model_large.train(data="uk_pest_dataset_24DEC.yaml", epochs=400, imgsz=512, batch=48, name= "uk_pest_24DEC_tiny")
#     model_large = YOLO(model_list[3])
#     model_large.train(data="uk_pest_dataset_01JAN.yaml", epochs=400, imgsz=512, batch=48, name= "uk_pest_01JAN_tiny")

#     model_large = YOLO(model_list[1])
#     model_large.train(data="uk_pest_dataset_24DEC.yaml", epochs=200, imgsz=640, batch=16, name= "uk_pest_24DEC_medium")
#     model_large = YOLO(model_list[4])
#     model_large.train(data="uk_pest_dataset_01JAN.yaml", epochs=200, imgsz=640, batch=16, name= "uk_pest_01JAN_medium")


#     model_large = YOLO(model_list[2])
#     model_large.train(data="uk_pest_dataset_24DEC.yaml", epochs=100, imgsz=640, batch=2, name= "uk_pest_24DEC_extra")
#     model_large = YOLO(model_list[5])
#     model_large.train(data="uk_pest_dataset_01JAN.yaml", epochs=100, imgsz=640, batch=2, name= "uk_pest_01JAN_extra")
