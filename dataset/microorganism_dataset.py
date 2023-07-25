import pandas
from build_annotation import SetAnnotation
import numpy as np
import json

def convert_to_xml():
    train_path = "F:\\nematoda\\Microorganism\\Train_Object.csv"
    test_path = "F:\\nematoda\\Microorganism\\Tests_Object.csv"

    train_dataset = pandas.read_csv(train_path)
    train_annotation = SetAnnotation("F:\\pest_data\\Builted_Dataset_In_2022\\Annotated_Data\\0_0.xml", "F:\\nematoda\\Microorganism\\Train set", "F:\\nematoda\\Microorganism\\Train set", ["Nematode"])

    bboxes = []  
    file_name = "" 
    for idx, data in train_dataset.iterrows():
        print(data["filename"], data["region_count"], data["region_shape_attributes"])

        region_shape_attributes = json.loads(data["region_shape_attributes"])

        if file_name == data["filename"]:
            bboxes.append([region_shape_attributes["x"], region_shape_attributes["y"], region_shape_attributes["x"]+region_shape_attributes["width"], region_shape_attributes["y"]+region_shape_attributes["height"], 0, 0])
            train_annotation(data["filename"].split(".")[0], [1392,1040], np.array(bboxes))
        else:
            file_name = data["filename"]
            bboxes = [[region_shape_attributes["x"], region_shape_attributes["y"], region_shape_attributes["x"]+region_shape_attributes["width"], region_shape_attributes["y"]+region_shape_attributes["height"], 0, 0]]
            train_annotation(data["filename"].split(".")[0], [1392,1040], np.array(bboxes))

    test_dataset = pandas.read_csv(test_path)
    val_annotation = SetAnnotation("F:\\pest_data\\Builted_Dataset_In_2022\\Annotated_Data\\0_0.xml", "F:\\nematoda\\Microorganism\\Tests set", "F:\\nematoda\\Microorganism\\Tests set", ["Nematode"])

    bboxes = []  
    file_name = ""
    for idx, data in test_dataset.iterrows():
        print(data["filename"], data["region_count"], data["region_shape_attributes"])

        region_shape_attributes = json.loads(data["region_shape_attributes"])

        if file_name == data["filename"]:
            bboxes.append([region_shape_attributes["x"], region_shape_attributes["y"], region_shape_attributes["x"]+region_shape_attributes["width"], region_shape_attributes["y"]+region_shape_attributes["height"], 0,0])
            val_annotation(data["filename"].split(".")[0], [2584,1936], np.array(bboxes))
        else:
            file_name = data["filename"]
            bboxes = [[region_shape_attributes["x"], region_shape_attributes["y"], region_shape_attributes["x"]+region_shape_attributes["width"],region_shape_attributes["y"]+region_shape_attributes["height"], 0, 0]]
            val_annotation(data["filename"].split(".")[0], [2584,1936], np.array(bboxes))


if __name__ == "__main__":
    convert_to_xml()
