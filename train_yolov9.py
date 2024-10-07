from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def training_original_yolo(dataset_yaml, folder_name):
    model_list = ["yolov9t.pt","yolov9m.pt"]

    model_tiny = YOLO(model_list[0])
    model_tiny.train(data=dataset_yaml, epochs=200, imgsz=640, batch=8, name= f"{folder_name}_tiny")

    model_medium = YOLO(model_list[1])
    model_medium.train(data=dataset_yaml, epochs=200, imgsz=640, batch=4, name= f"{folder_name}_medium")


if __name__ == "__main__":
    dataset_yaml = "uk_pest_dataset_18SEP24_all_insect"
    training_original_yolo(dataset_yaml+".yaml","yolov9_640_"+dataset_yaml)

    dataset_yaml = "uk_pest_dataset_18SEP24_pest_only"
    training_original_yolo(dataset_yaml+".yaml","yolov9_640_"+dataset_yaml)

