import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import os
import csv

from openai import OpenAI

def get_bounding_box(xml_annotation):
    voc_annotation_file = open(xml_annotation, encoding='utf-8')
    tree = ET.parse(voc_annotation_file)
    root = tree.getroot()

    bounding_boxes = []

    for obj in root.iter('object'):
        # cls_id = 0
        xmlbox = obj.find('bndbox')
        b = [int(xmlbox.find('xmin').text)-50, int(xmlbox.find('ymin').text)-50, int(xmlbox.find('xmax').text)+50, int(xmlbox.find('ymax').text)+50]
        bounding_boxes.append(b)

    return bounding_boxes

def get_description_for_images(image_path,xml_annotation, client):

    objects_boxes = get_bounding_box(xml_annotation)

    image = Image.open(image_path)

    object_features = []

    for box in objects_boxes:
        print(box)
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(image.size[0], box[2])
        box[3] = min(image.size[1], box[3]) 
        subimage = np.array(image)[min(box[1],box[3]):max(box[1],box[3]),min(box[0],box[2]):max(box[0],box[2])]

        feature = call_gpt_with_cutted_object(subimage, client)

        object_features.append(feature)

        # subimage = image.crop()
        # plt.subplot(1,2,1)
        # plt.imshow(image)
        # plt.subplot(1,2,2)
        # plt.imshow(subimage)
        # plt.show()

    return object_features

def call_gpt_with_cutted_object(image, client:OpenAI):
    # https://platform.openai.com/docs/guides/vision
    response = client.chat.completions.create(
        model=""
    )



if __name__ == "__main__":

    image_folder = "F:\\pest_data\\Multitask_or_multimodality\\annotated_images"

    objects_dict = {}

    client = OpenAI()

    for root, folders, files in os.walk(image_folder):
        for file in files:
            if file.split(".")[-1] == "JPG":
                print(file)
                object_features = get_description_for_images(os.path.join(root, file), os.path.join(root, file.split(".")[0]+".xml"), client)

                objects_dict[file.split(".")[0]] = object_features

    w = csv.DictWriter(open(os.path.join(image_folder,"gpt_annotatation.csv"),'w'))
    w.writerows(objects_dict.items())
