from ultralytics import YOLO
import cv2
import os 
import tqdm
import time
import seaborn

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

def evaluation_by_sourcecode(model_path):
    model = YOLO(model_path)

    metrics = model.val("nematodes.yaml")

    # model.val( save_json=True, save_dir="val")

    print(metrics.box.speed)

    print(metrics.box.map)

    print(metrics.box.p)

    print(metrics.box.r)

    print(metrics.box.f1)

    print(metrics.box.ap50())

if __name__ == "__main__":
    model_path = "runs\\detect\\train\\weights\\best.pt"
    dataset_dir = "F:\\nematoda\\nemadote_detection"

    evaluation_by_sourcecode(model_path)

    # prediction(model_path, dataset_dir)
    # prediction_speed(model_path, dataset_dir)