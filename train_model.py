from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse

def training_original_yolo(dataset_yaml, folder_name):
    model_list = ["runs\\detect\\YOLO_25JUN24_ALL_INSECT_tiny4\\weights\\best.pt","runs\\detect\\YOLO_25JUN24_ALL_INSECT_medium2\\weights\\best.pt","runs\\detect\\YOLO_25JUN24_ALL_INSECT_extra\\weights\\best.pt"]

    model_tiny = YOLO(model_list[0])
    model_tiny.train(data=dataset_yaml, epochs=200, imgsz=640, batch=16, name= f"{folder_name}_tiny")

    model_medium = YOLO(model_list[1])
    model_medium.train(data=dataset_yaml, epochs=200, imgsz=640, batch=8, name= f"{folder_name}_medium")

    model_large = YOLO(model_list[2])
    model_large.train(data=dataset_yaml, epochs=200, imgsz=640, batch=4, name= f"{folder_name}_extra")


def training_yolov10(dataset_yaml):
    model_list = ["yolov10n.pt","yolov10m.pt"]

    model_tiny = YOLO(model_list[0])
    model_tiny.train(data=dataset_yaml, epochs=200, imgsz=640, batch=8, name= f"{dataset_yaml.split('.')[0]}_{model_list[0].split('.')[0]}_640")

    model_medium = YOLO(model_list[1])
    model_medium.train(data=dataset_yaml, epochs=200, imgsz=640, batch=4, name= f"{dataset_yaml.split('.')[0]}_{model_list[0].split('.')[0]}_640")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO model")

    # 添加参数
    parser.add_argument('--yaml_path', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--models', nargs='+', required=True, help='List of models for training (e.g. yolov5s, yolov5m)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')

    return parser.parse_args()

def train_model(yaml_path, models, epochs, img_size, batch_size):

    for model in models:
        model = YOLO(model)
        model.train(data=yaml_path, epochs=epochs, imgsz=img_size, batch=batch_size, name=f"{model.split('.')[0]}_{img_size}_{yaml_path.split('.')[0]}")

if __name__ == "__main__":

    args = parse_args()

    # training_yolov10("uk_pest_dataset_18SEP24_all_insect.yaml")

    train_model(
        yaml_path=args.yaml_path, 
        models=args.models, 
        epochs=args.epochs, 
        img_size=args.img_size, 
        batch_size=args.batch_size)