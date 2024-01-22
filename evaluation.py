from ultralytics import YOLO
import cv2
import os 
import tqdm
import time
import seaborn
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def prediction(model_path, dataset_dir):
    test_image_dir = f"{dataset_dir}\\images\\test"

    model = YOLO(model_path)

    prediction_save_path = f"{dataset_dir}\\test_prediction"
    if not os.path.exists(prediction_save_path):
        os.mkdir(prediction_save_path)

    for root, folders, files in os.walk(test_image_dir):
        with tqdm.tqdm(total=len(files)) as tbar:  
            for file in files:
                results = model(os.path.join(root,file))
                #  format x1,y1,x2,y2,conf,cls
                prediction =  results[0].boxes
                print(len(prediction.cpu().numpy()))
                print(results)
                
                break
                tbar.update(1)

def calculate_metrics(dataset_dir):
    ground_truth_label_dir = f"{dataset_dir}\\labels\\test"

    tp=0
    fp=0
    tn=0
    fn=0

def prediction_speed(model_path, dataset_dir):

    test_image_dir = f"{dataset_dir}\\images\\test"

    model = YOLO(model_path)

    for root, folders, files in os.walk(test_image_dir):
        # with tqdm.tqdm(total=len(files)) as tbar:  
            sample_numbers = len(files)
            start_time = time.time()
            for file in files:
                results = model(os.path.join(root,file))
                #  format x1,y1,x2,y2,conf,cls
                # tbar.update(1)
            end_time = time.time()
    
    print(f"{sample_numbers} images, spend {end_time - start_time}s, each image {(end_time - start_time)/sample_numbers}")

def evaluation_by_sourcecode(model_name, dataset_yaml_name):
    model = YOLO(f"runs\\detect\\{model_name}\\weights\\best.pt")

    # metrics = model.val("nematodes_test.yaml",save_json=True)
    metrics = model.val(f"{dataset_yaml_name}.yaml",save_json=True, name=f"{model_name}_val_on_{dataset_yaml_name}")

    # model.val("nematodes.yaml", save_json=True, save_dir="val")

    print(metrics)

    print(metrics.box.map)

    print(metrics.box.p)

    print(metrics.box.r)

    print(metrics.box.f1)

    print(metrics.box.ap50, np.mean(metrics.box.ap50))

if __name__ == "__main__":
    model_path = "runs\\detect\\uk_pest_24Dec_tiny\\weights\\best.pt"
    dataset_yaml_path = "uk_pest_dataset_25DEC.yaml"
    # model_path = "runs\\detect\\train\\weights\\last.pt"
    # dataset_dir = "F:\\nematoda\\nemadote_detection"
    # model_name = ["uk_pest_24Dec_tiny","uk_pest_24DEC_tiny2","uk_pest_24Dec_medium","uk_pest_24DEC_medium2","uk_pest_24Dec_extra","uk_pest_24DEC_extra2","uk_pest_25DEC_extra","uk_pest_25DEC_medium"]
    model_name = ["uk_pest_25DEC_tiny","uk_pest_25DEC_extra","uk_pest_25DEC_medium"]
    dataset_yaml_name = ["uk_pest_dataset_25DEC", "uk_pest_dataset_24DEC"]

    for model in model_name:
        for dataset in dataset_yaml_name:
            evaluation_by_sourcecode(model, dataset)

    # prediction(model_path, dataset_dir)
    # prediction_speed(model_path, dataset_dir)