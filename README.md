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

### 1.5 Automatic Annotation Generation

```
python dataset/build_annotation.py
```

## 2 Useful external link

- [Yolov8 Docs](https://docs.ultralytics.com/)
- [Yolov5 Docs](https://github.com/ultralytics/yolov5)
- [Pretrained Yolov8 Model](https://github.com/ultralytics/ultralytics)


## Update Model Structure

1. Build Models Config Yaml file

```
# Ultralytics YOLO 🚀, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # scales module repeats
width_multiple: 0.25  # scales convolution channels

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]   # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 3, CBAM, [128]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 3, CBAM, [256]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 3, CBAM, [512]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 3, CBAM, [1024]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 8], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 5], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

2. Add additional modules to the ../python_env/yolov8/Lib/site-packages/ultralytics/nn/modules.py

3. Modify task.py for loading modules function parameters. ../python_env/yolov8/lib/site-packages/ultralytics/nn/tasks.py
    - Import package first 
    ```
    from ultralytics.nn.modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, Classify,
                                    Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Ensemble, Focus,
                                    GhostBottleneck, GhostConv, Segment, CBAM)
    ``` 
    - Updata modules function parameters loading function
    ```
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in (nn.BatchNorm2d, CBAM):
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(args[2] * gw, 8)
        else:
            c2 = ch[f]
    ```


## Update Parameter Loading Methods

In the .py file ../python_env/yolov8/Lib/site-packages/ultralytics/yolo/utils/torch_utils.py, update intersect_dicts(da, db, exclude=()) function by the following code

```

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    # return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


    db_index = 0
    db_max_layer = int(list(db.keys())[-1].split(".")[1])
    results = {}

    for k,v in da.items():
        name_list = k.split(".")
        db_temp_index = max(db_index, int(name_list[1])) 
        name_list[1] = str(db_temp_index)
        db_k = ".".join(name_list) 

        while (not db_k in db) and (db_temp_index<db_max_layer):
            db_temp_index += 1
            name_list[1] = str(db_temp_index)
            db_k = ".".join(name_list)

        if (db_k in db) and (not db_k in results) and (v.shape == db[db_k].shape):
            results[db_k]=v
            db_index = db_temp_index
            print("Find: ", k, ":", db_k)
            continue

        print("Not find: ",k)

    return results
```

## Update data augmentation method

## Update lost function

## Update to multimodal

## Add explanation function
