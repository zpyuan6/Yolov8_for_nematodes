# Yolov8_for_nematodes

This is a project for nematode detection based on [Yolov8](https://github.com/ultralytics/ultralytics). 

## 1. How to use

See below for quickstart installation and usage examples for model training, object detection, and deployment.

Please install all packages including all [requirements.txt](./requirements.txt) first in a [Python>=3.7](https://www.python.org/) with [Pytorch>=1.7](https://pytorch.org/get-started/previous-versions/). 
I recommend using [Conda](https://www.anaconda.com/) to manage your Python environment.

```
git clone https://github.com/zpyuan6/Yolov8_for_nematodes.git
cd Yolov8_for_nematodes
conda create -n yolov8 python=3.8.16 
conda activate yolov8
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install scikit-image
pip install -r requirements.txt
```

### 1.1 Object detection
Yolov8 DOCS for prediction https://docs.ultralytics.com/modes/predict/

```
python detection.py
```

### 1.2 Model training
Yolov8 DOCS for training https://docs.ultralytics.com/modes/train/

The first stage for model training is dataset preparation.
Section 1.1 in [Yolov5 wiki](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) presents how you should prepare the data set manually in detail.
Briefly, you are required to do the following stages.

1. Create dataset.yaml file as [nematodes.yaml](./nematodes.yaml).
2. Create Labels in Yolo format in .txt file as follows.
    ```
    0 0.4766768292682927 0.7664366883116883 0.0125 0.0166396103896103880
    0 0.3463414634146341 0.5456574675324676 0.006097560975609756 0.011769480519480520
    ```
    Each line is (Class index) (Centre point x position) (Centre point y position) (Bounding box width) (Bounding box height). Note that each value in is a scale, not pixel value.
3. Organise dataset directories as follows.
    ```
    ../datasets/coco128/images/train/im0.jpg  # image
    ../datasets/coco128/labels/train/im0.txt  # label
    ```

Set parameters and training.
```
python train_model.py
```

### 1.3 Model evaluation

```
python evuation.py
```

### 1.4 Model deployment
Yolov8 DOCS for deployment https://docs.ultralytics.com/modes/export/

## 2 Useful external link

- [Yolov8 Docs](https://docs.ultralytics.com/)
- [Yolov5 Docs](https://github.com/ultralytics/yolov5)
- [Pretrained Yolov8 Model](https://github.com/ultralytics/ultralytics)
