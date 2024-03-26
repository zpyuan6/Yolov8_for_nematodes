from sklearn.cluster import KMeans
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import match_descriptors, plot_matches, SIFT
import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt

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

def cut_object(image_path, xml_path, object_index):
    objects_boxes = get_bounding_box(xml_path)

    image = Image.open(image_path)

    box = objects_boxes[object_index]

    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(image.size[0], box[2])
    box[3] = min(image.size[1], box[3]) 
    subimage = np.array(image)[min(box[1],box[3]):max(box[1],box[3]),min(box[0],box[2]):max(box[0],box[2])]

    return subimage



def get_object_feature(image_path,xml_annotation):

    objects_boxes = get_bounding_box(xml_annotation)

    image = Image.open(image_path)

    object_features = []

    descriptor_extractor = SIFT()

    for box in objects_boxes:
        print(box)
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(image.size[0], box[2])
        box[3] = min(image.size[1], box[3]) 
        subimage = np.array(image)[min(box[1],box[3]):max(box[1],box[3]),min(box[0],box[2]):max(box[0],box[2])]

        img = rgb2gray(subimage)
        img = resize(img, (100,100), anti_aliasing=True)

        descriptor_extractor.detect_and_extract(img)
        feature = descriptor_extractor.descriptors
        if feature.shape[0]<20:
            feature = np.pad(feature, ((0,20-feature.shape[0]),(0,0)), 'constant', constant_values=(0))

        feature = feature[0:20,:].flatten()

        object_features.append(feature)

        # subimage = image.crop()
        # plt.subplot(1,2,1)
        # plt.imshow(image)
        # plt.subplot(1,2,2)
        # plt.imshow(subimage)
        # plt.show()

    return object_features

def cluster(objects_dict:dict):

    objects_list = []

    for image_id in objects_dict.keys():
        objects_list += objects_dict[image_id]

    print(np.array(objects_list).shape)

    X = np.array(objects_list)

    kmeans = KMeans(n_clusters=30, random_state=0, n_init="auto").fit(X)

    print(kmeans.labels_)

    pickle.dump(kmeans, open("F:\\pest_data\\Multitask_or_multimodality\\annotated_images\\cluster_model.txt",'wb'))

    pickle.dump(objects_dict, open("F:\\pest_data\\Multitask_or_multimodality\\annotated_images\\objects__features_dict.txt",'wb'))

def visualize_clustering_results(image_folder, cluster_model_path, objects_dict_path):

    cluster_model = pickle.load(open(cluster_model_path,'rb'))
    objects_feature_dict = pickle.load(open(objects_dict_path,'rb'))

    # print(objects_feature_dict)
    print(cluster_model.labels_)
    print(cluster_model.cluster_centers_)

    objects_list = []
    image_list = []
    object_id_list = []

    for image_id in objects_feature_dict.keys():
        objects_list += objects_feature_dict[image_id]

        for i in range(len(objects_feature_dict[image_id])):
            image_list.append(image_id)
            object_id_list.append(i)

    object_cluster_dict = {}

    for i,label in enumerate(cluster_model.labels_):
        if label in object_cluster_dict:
            object_cluster_dict[label].append(i)
        else:
            object_cluster_dict[label] = [i]

    for label in object_cluster_dict.keys():
        one_class_features = []
        for i, object_index in enumerate(object_cluster_dict[label]) :
            plt.subplot(1,5,i+1)
            plt.imshow(cut_object(os.path.join(image_folder,image_list[object_index]+".JPG") , os.path.join(image_folder,image_list[object_index]+".xml"), object_id_list[object_index]))

            if i==4:
                break
        plt.suptitle(label)
        plt.show()





if __name__ == "__main__":
    image_folder = "F:\\pest_data\\Multitask_or_multimodality\\annotated_images"

    objects_dict = {}

    for root, folders, files in os.walk(image_folder):
        for file in files:
            if file.split(".")[-1] == "JPG":
                print(file)
                object_features = get_object_feature(os.path.join(root, file), os.path.join(root, file.split(".")[0]+".xml"))

                objects_dict[file.split(".")[0]] = object_features

    cluster(objects_dict)

    visualize_clustering_results(image_folder,"F:\\pest_data\\Multitask_or_multimodality\\annotated_images\\cluster_model.txt", "F:\\pest_data\\Multitask_or_multimodality\\annotated_images\\objects__features_dict.txt")








