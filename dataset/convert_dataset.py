import os
import shutil
import xml.etree.ElementTree as ET
import random

CLASSES = ["11121","11122"]

def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 

def convert(size,box):

    x = (box[0] + box[2])/2.0 - 1
    y = (box[1] + box[3])/2.0 - 1 
    x = x/float(size[0])
    y = y/float(size[1])

    w = (box[3] - box[1])/float(size[0])
    h = (box[2] - box[0])/float(size[1])

    return x,y,w,h

def convert_annotation(annotation_file_path, list_file):
    voc_annotation_file = open(annotation_file_path, encoding='utf-8')
    tree = ET.parse(voc_annotation_file)
    root = tree.getroot()

    width = root.find('size')[0].text
    height = root.find('size')[1].text

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # if cls not in CLASSES:
        #     continue

        # cls_id = classes.index(cls)
        cls_id = 0
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        bb = convert((width,height),b)
        list_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def convert_dataset_from_voc_to_yolo(original_path, target_path):
    train_list_file_path = os.path.join(original_path, "ImageSets\\Main\\train.txt")
    test_list_file_path = os.path.join(original_path, "ImageSets\\Main\\test.txt")
    val_list_file_path = os.path.join(original_path, "ImageSets\\Main\\val.txt")

    images_folder = os.path.join(original_path,"JPEGImages")
    labels_folder = os.path.join(original_path,"YOLOLabels")

    train_yolo_image_folder = os.path.join(target_path, "images\\train")
    test_yolo_image_folder = os.path.join(target_path, "images\\test")
    val_yolo_image_folder = os.path.join(target_path, "images\\val")

    train_yolo_label_folder = os.path.join(target_path, "labels\\train")
    test_yolo_label_folder = os.path.join(target_path, "labels\\test")
    val_yolo_label_folder = os.path.join(target_path, "labels\\val")

    check_and_create_folder(train_yolo_image_folder)
    check_and_create_folder(test_yolo_image_folder)
    check_and_create_folder(val_yolo_image_folder)
    check_and_create_folder(train_yolo_label_folder)
    check_and_create_folder(test_yolo_label_folder)
    check_and_create_folder(val_yolo_label_folder)

def convert_xml_to_yolo(source_folder, target_folder):
    train_val = 0.8

    train_yolo_image_folder = os.path.join(target_folder, "images\\train")
    val_yolo_image_folder = os.path.join(target_folder, "images\\val")

    train_yolo_label_folder = os.path.join(target_folder, "labels\\train")
    val_yolo_label_folder = os.path.join(target_folder, "labels\\val")

    check_and_create_folder(train_yolo_image_folder)
    check_and_create_folder(val_yolo_image_folder)
    check_and_create_folder(train_yolo_label_folder)
    check_and_create_folder(val_yolo_label_folder)

    for root, folders, files in os.walk(source_folder):
        for file in files:
            file_name,file_type = os.path.splitext(file)
            if not file_type == ".xml":
                ro = random.random()
                if ro < train_val:
                    if file_type == ".JPG":
                        shutil.copy(os.path.join(root,file),os.path.join(train_yolo_image_folder,file_name+".jpg"))
                    else:
                        shutil.copy(os.path.join(root,file),os.path.join(train_yolo_image_folder,file))

                    annotation_file = open(os.path.join(train_yolo_label_folder,file_name+".txt"), 'w', encoding='utf-8')

                    convert_annotation(os.path.join(root,file_name+".xml"),annotation_file)

                    annotation_file.write('\n')

                    annotation_file.close()
                else:
                    if file_type == ".JPG":
                        shutil.copy(os.path.join(root,file),os.path.join(val_yolo_image_folder,file_name+".jpg"))
                    else:
                        shutil.copy(os.path.join(root,file),os.path.join(val_yolo_image_folder,file))

                    annotation_file = open(os.path.join(val_yolo_label_folder,file_name+".txt"), 'w', encoding='utf-8')

                    convert_annotation(os.path.join(root,file_name+".xml"),annotation_file)

                    annotation_file.write('\n')

                    annotation_file.close()



def copy_annotation(yolo_folder,voc_folder):
    annotation_folder = os.path.join(voc_folder, "YOLOLabels")
    train_images = os.path.join(yolo_folder,"images\\train")
    val_images = os.path.join(yolo_folder,"images\\val")

    train_labels = os.path.join(yolo_folder, "labels\\train")
    val_labels = os.path.join(yolo_folder, "labels\\val")
    check_and_create_folder(train_labels)
    check_and_create_folder(val_labels)

    for root, folders, files in os.walk(train_images):
        for file in files:
            image_id = file.split(".")[0]
            source_annotation_file = os.path.join(annotation_folder, f"{image_id}.txt")
            target_annotation_file = os.path.join(train_labels, f"{image_id}.txt")
            shutil.copy(source_annotation_file, target_annotation_file)



    for root, folders, files in os.walk(val_images):
        for file in files:
            image_id = file.split(".")[0]
            source_annotation_file = os.path.join(annotation_folder, f"{image_id}.txt")
            target_annotation_file = os.path.join(val_labels, f"{image_id}.txt")
            shutil.copy(source_annotation_file, target_annotation_file)


if __name__ == "__main__":
    voc_path = "F:\\Pest\\pest_data\\Pest_Dataset_2023"
    yolo_path = "F:\\Pest\\pest_data\\YOLO_2023"
    # copy_annotation(yolo_path, voc_path)
    convert_xml_to_yolo(voc_path, yolo_path)