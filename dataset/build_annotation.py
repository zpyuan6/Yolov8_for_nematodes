import xml.etree.ElementTree as ET
import pickle
import os
from collections import OrderedDict
from os import listdir, getcwd
from os.path import join
import copy
import torch
from ultralytics import YOLO


# ['Meloidogyne hapla', 'Globodera pallida', 'Pratylenchus', 'Ditylenchus']

class SetAnnotation():
    def __init__(self,in_file_xml,in_dir_img,out_dir_xml,classes=['Pest', 'Spider', 'Fly', 'Snail','Aphids','Slug','Beetle']):
        # assert(in_dir_xml!=out_dir_xml)
        self.in_file_xml = in_file_xml
        self.in_dir_img = in_dir_img
        self.out_dir_xml = out_dir_xml
        self.classes = classes

    def __call__(self,image_name,imagesize,pred):
        global data_root
        #确定文件的输入输出
        filepath = f'{self.in_dir_img}/{image_name}.jpg'
        in_file = open(self.in_file_xml, encoding='utf-8')
        out_file = f'{self.out_dir_xml}/{image_name}.xml'
        #获取文件的层次结构
        tree=ET.parse(in_file)
        root = tree.getroot()  #获取root节点，即1.3中的annotation节点
        folder = root.find('folder')
        folder.text = self.in_dir_img
        filename = root.find('filename')
        filename.text = image_name
        path = root.find('path')
        path.text = filepath
        size = root.find('size') #获取size节点
        size.find('width').text = f'{imagesize[0]}'#获取宽度节点信息
        size.find('height').text=  f'{imagesize[1]}'#获取高度节点信息
        if len(imagesize) == 3:
            size.find('depth').text =  f'{imagesize[2]}'
        objects = root.find('object')

        if pred.shape[0] == 0:
            return

        nums_new_object = pred.shape[0] - len(root.findall('object'))

        if nums_new_object > 0:
            for i in range(nums_new_object):
                # Create a copy
                obj = copy.deepcopy(objects)
                # Append the copy
                root.append(obj)
        elif nums_new_object < 0:
            for obj in root.findall('object'):
                root.remove(obj)
                nums_new_object+=1
                if nums_new_object==0:
                    break

        for i, obj in enumerate(root.iter('object')):  #迭代获取所有的object节点
            print(pred[i])
            cls = self.classes[int(pred[i][5])]
            obj.find('name').text = cls          #获取name节点信息，即bbox的类别信息
            xmlbox = obj.find('bndbox')          #获取bbox的左上角点与右下角点坐标信息
            if int(pred[i][0]) != 0:
                xmlbox.find('xmin').text = f'{int(pred[i][0])}'
                xmlbox.find('ymin').text = f'{int(pred[i][1])}'
                xmlbox.find('xmax').text = f'{int(pred[i][2])}'
                xmlbox.find('ymax').text = f'{int(pred[i][3])}'
            else:
                xmlbox.find('xmin').text = f'{int(pred[i][0]* imagesize[0]) }'
                xmlbox.find('ymin').text = f'{int(pred[i][1]* imagesize[1]) }'
                xmlbox.find('xmax').text = f'{int(pred[i][2]* imagesize[0]) }'
                xmlbox.find('ymax').text = f'{int(pred[i][3]* imagesize[1]) }'
        # print(out_file)
        tree.write(out_file,encoding='utf-8')



if  __name__ == '__main__':
    path = "F:\\pest_data\\Multitask_or_multimodality\\unannotated_images"
    model_path = "runs\\pest_uk_medium_07_062\\weights\\best.pt"
    # class_list = ["Meloidogyne", "Cyst", "Globodera", "Pratylenchus", "PCN j2", "Ditylenchus"]
    class_list = ["Insecta", "Insecta", "Insecta", "Insecta", "Insecta", "Insecta", "Insecta", "Insecta", "Insecta", "Insecta"]

    model = YOLO(model_path)

    for root, folders, files in os.walk(path):
        if root!=path:
            break
        for file in files:
            if file.split(".")[-1] == "JPG" or file.split(".")[-1] == "jpg" or file.split(".")[-1] == "JPEG" or file.split(".")[-1] == "tif":
                setANN = SetAnnotation("F:\\pest_data\\Multitask_or_multimodality\\annotated_images\\0x1a37ffe63075ee0.xml",os.path.join(root,file), root, class_list)
                pred = model(os.path.join(root,file), conf=0.585)
                # print(pred[0])
                setANN('.'.join(file.split('.')[0:-1]), [pred[0].orig_shape[1],pred[0].orig_shape[0]],pred[0].boxes.data.cpu().numpy())