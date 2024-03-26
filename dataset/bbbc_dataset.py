import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as T
import json
from build_annotation import SetAnnotation
import random
import shutil        

def show_image():
    img = io.imread("F:\\nematoda\\BBBC010\\1649_1109_0003_Amp5-1_B_20070424_A01_w2_15ADF48D-C09E-47DE-B763-5BC479534681.tif")
    print(img.shape)
    print(np.min(img),np.max(img))

    # img = (img - np.min(img))/(np.max(img)-np.min(img))
    # print(img)

    plt.imshow(img)
    plt.show()

def convert_tif_to_png():
    path = "F:\\nematoda\\BBBC010"

    transform=T.ToPILImage()

    for root, folders, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "tif":
                if file.split(".")[0].split("_")[7] == "w1":
                    img_path = os.path.join(root, file.split(".")[0].split("_")[6]+".jpg")
                    img = io.imread(os.path.join(root,file)).astype(np.uint8)
                    data = Image.fromarray(img)
                    data.save(img_path)

def convert_img_to_bbox(img_path):
    img = io.imread(img_path)
    print(img.shape, np.min(img), np.max(img))

    x = np.where(np.max(img,axis=0)==np.max(img))[0]
    x1,x2 = x[0], x[-1]
    y = np.where(np.max(img,axis=1)==np.max(img))[0]
    y1,y2 = y[0], y[-1]
                    # print(x1,x2,y1,y2)

    print([int(x1),int(y1),int(x2),int(y2)])
    return [int(x1),int(y1),int(x2),int(y2)]
    


def convert_instance_segmentation_to_object_detection():

    path = "F:\\nematoda\\BBBC010"
    label_path = "F:\\nematoda\\BBBC010\\labels"
    labels= {}

    for root, folders, files in os.walk(label_path):
        for file in files:
            label = file.split("_")[0]
            if label in labels:
                labels[label].append(convert_img_to_bbox(os.path.join(root,file))) 
            else:
                labels[label] = [convert_img_to_bbox(os.path.join(root,file))]

    for label in labels.keys():
        bboxes = labels[label]
        annotation_path= os.path.join(path, label+".txt")
        annotation_file = open(annotation_path,"w")

        setANN = SetAnnotation("F:\\pest_data\\Builted_Dataset_In_2022\\Annotated_Data\\0_0.xml", path, path, ["Live Nematode","Dead Nematode"])

        writing_content = []
        numpy_annotation = []
        for bounding_box in bboxes:
            clabel = 1 if int(label[1:]) > 12 else 0

            writing_content.append(f"{clabel} {(bounding_box[0]+bounding_box[2])/2/690} {(bounding_box[1]+bounding_box[3])/2/520} {(bounding_box[2]-bounding_box[0])/690} {(bounding_box[3]-bounding_box[1])/520}")

            numpy_annotation.append([bounding_box[0],bounding_box[1], bounding_box[2], bounding_box[3], 0,clabel])

        bboxes = np.array(numpy_annotation)
        setANN(label,[696, 520], bboxes)

        annotation_file.writelines(line + '\n' for line in writing_content)
        annotation_file.close()
        
    
    # j = json.dumps(labels)
    # f = open(f"F:\\nematoda\\BBBC010\\bounding_boxes.json","w")
    # f.write(j)
    # f.close()  

# def convert_yolo_txt_to_xml():
#     path = "F:\\nematoda\\BBBC010"

#     for root, folders, files in os.walk(path):
#         for file in files:
#             if file.split(".")[-1] == "txt":
#                 print(os.path.join(root,file))
#                 setANN = SetAnnotation("F:\\pest_data\\Builted_Dataset_In_2022\\Annotated_Data\\0_0.xml",os.path.join(root,file), root, ["Live Nematode","Dead Nematode"])
#                 bboxs = np.loadtxt(os.path.join(root,file))
#                 print(bboxs)
#                 setANN(file.split('.')[0],[696, 520], bboxs)

def random_split_dataset():
    path = "F:\\nematoda\\BBBC010"
    yolo_path = "F:\\nematoda\\BBBC010\\YOLO"
    if not os.path.exists(yolo_path):
        os.makedirs(os.path.join(yolo_path,"images","train"))
        os.makedirs(os.path.join(yolo_path,"images","val"))
        os.makedirs(os.path.join(yolo_path,"labels","train"))
        os.makedirs(os.path.join(yolo_path,"labels","val"))

    images_id = []
    for root,folder, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "jpg":
                images_id.append(file.split(".")[0])

    training_set = random.sample(images_id, int(0.9*len(images_id)))
    val_set = list(set(images_id).difference(set(training_set)))

    for image_id in training_set:
        shutil.copy2(os.path.join(path, image_id+".jpg"), os.path.join(yolo_path,"images","train",image_id+".jpg"))
        shutil.copy2(os.path.join(path, image_id+".txt"), os.path.join(yolo_path,"labels","train",image_id+".txt"))

    for image_id in val_set:
        shutil.copy2(os.path.join(path, image_id+".jpg"), os.path.join(yolo_path,"images","val",image_id+".jpg"))
        shutil.copy2(os.path.join(path, image_id+".txt"), os.path.join(yolo_path,"labels","val",image_id+".txt"))


if __name__=="__main__":
    # show_image()
    # convert_tif_to_png()
    # convert_img_to_bbox("F:\\nematoda\\BBBC010\\labels\\A01_01_ground_truth.png")
    # convert_instance_segmentation_to_object_detection()
    random_split_dataset()