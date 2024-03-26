import os
import shutil
from  clean_and_stat_pest_dataset import aHash
from matplotlib.pyplot import imread
from PIL import ImageChops,Image, ExifTags
import json
import xml.etree.ElementTree as ET
from skimage import io
from datetime import datetime

from convert_dataset import convert

IMAGE_FILE_TYPE = [".jpeg",".jpg",".JPG",".JPEG",".png",".PNG",".tiff",".tif"]

def save_to_file(path:str, obj:object):
    file = open(path, "w")
    obj = json.dumps(obj)
    file.write(obj)
    file.close()

def load_json(path:str):
    with open(path,'r') as file:
        json_object = json.load(file)

    print(json_object, type(json_object))

def create_multitask_or_multimodel_dataset_from_xml(original_path:list, target_path:str):
    all_image_hash = []
    annotated_id = []
    unannotated_id = []
    duplicate_image = {}

    annotated_path = os.path.join(target_path, "annotated_images")
    unannotated_path = os.path.join(target_path, "unannotated_images")

    if not os.path.exists(annotated_path):
        os.makedirs(annotated_path)
    if not os.path.exists(unannotated_path):
        os.makedirs(unannotated_path)

    for folder_path in original_path:
        for root, folder, files in os.walk(folder_path):
            for file in files:
                image_id, file_type = os.path.splitext(file)
                print(image_id, file_type)

                if file_type in IMAGE_FILE_TYPE:
                    # find annotation (xml file)
                    img = imread(os.path.join(root, file))
                    image_hash = aHash(img)
                    image_hash = hex(int(image_hash, 2))
                    # print(image_hash, type(image_hash))

                    if image_hash in all_image_hash:
                        if image_hash in duplicate_image:
                            duplicate_image[image_hash].append(os.path.join(root, file))
                        else:
                            duplicate_image[image_hash] = [os.path.join(root, file)]
                    else:
                        all_image_hash.append(image_hash)
                        if os.path.exists(os.path.join(root, f"{image_id}.xml")):
                            annotated_id.append(os.path.join(root, file))
                            if not os.path.exists(os.path.join(annotated_path ,f"{image_hash}.JPG")):
                                shutil.copyfile(os.path.join(root,file), os.path.join(annotated_path ,f"{image_hash}.JPG"))
                            if not os.path.exists(os.path.join(annotated_path ,f"{image_hash}.xml")):
                                shutil.copyfile(os.path.join(root, f"{image_id}.xml"), os.path.join(annotated_path, f"{image_hash}.xml"))
                        else:
                            unannotated_id.append(os.path.join(root, file))
                            if not os.path.exists(os.path.join(unannotated_path ,f"{image_hash}.JPG")):
                                shutil.copyfile(os.path.join(root,file), os.path.join(unannotated_path ,f"{image_hash}.JPG"))

    save_to_file(os.path.join(target_path, "all_image_hash.json"), all_image_hash)
    save_to_file(os.path.join(target_path, "annotated_id.json"), annotated_id)
    save_to_file(os.path.join(target_path, "unannotated_id.json"), unannotated_id)
    save_to_file(os.path.join(target_path, "duplicate_image.json"), duplicate_image)


# def test_img(path):
#     img = Image.open(path)
#     print(img)

loaded_pest = ['aphid', 'cabbage\\taphid', 'fly', 'pest', 'spider', 'beetle', 'Cabbage stem flea bettle', 'unknown', 'slug', 'Cabbage whitefly', 'Frog hopper (not pest)', 'Fly', 'Unidentifiable', 'Parasitic wasp', 'snail', 'Pest', 'Plant bug (Hemiptera) unknown', 'Beetle', 'Leaf miners', 'Chironomid midge (male)', 'Hemiptera (plant bug)', 'Male wasp(Social)', 'Diptera', 'Cabbage Whitefly', 'Fungus gnat (Mycetophilidae)']


first_level = ["diptera", "celyphidae", "mollusca", "hemiptera", "hymenoptera", "araneida", "unidentify"]
second_level = [["fly","leaf miners", "chironomid midge", "fungus gnat"],["beetle","cabbage stem flea bettle"],["snail","slug"],["aphid", "cabbage whitefly",'Frog hopper',"bug"],["parasitic wasp", "pale wasp"],["spider"]]

# import 
CATEGORY = ["insect", "fly","leaf miners", "chironomid midge", "fungus gnat","beetle","cabbage stem flea bettle","snail","slug","aphid", "cabbage whitefly",'Frog hopper',"bug","parasitic wasp", "male wasp", "spider"]

def format_multimodal_annotation(annotated_data_path):
    CROP_LIST = ["Wheat", "Radish", "Rapeseed", "Potato","Soybean"]
    PEST_LIST = []
    # CLASSES_MAP = [0.1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    CLASSES_MAP = [9,9,1,0,15,5, 6, 0, 8, 10, 11, 1, 0, 13, 7, 0, 12, 5, 2, 3, 12, 14, 1, 10, 4]

    DATA_FORMAT = '%Y:%m:%d %H:%M:%S'

    classes_dict = {}
    image_no_time= []

    for root, folders, files in os.walk(annotated_data_path):
        for file in files:
            image_id, file_type = os.path.splitext(file)
            if not (file_type == ".xml" or  file_type == ".txt"):
                image = Image.open(os.path.join(root, file))

                annotation_file = open(os.path.join(root, f"{image_id}.txt"), "w")
                
                if 306 in image.getexif():
                    photo_time = image.getexif()[306]
                    date_label = datetime.strptime(photo_time, DATA_FORMAT).strftime('%j')
                else:
                    # image_no_time.append(image_id)
                    if image_id == "0x70f0ecd896f7e2b4":
                        photo_time = "3/20/2023"
                        date_label = datetime.strptime(photo_time, "%m/%d/%Y").strftime('%j')
                    else:
                        photo_time = "4/20/2022"
                        date_label = datetime.strptime(photo_time, "%m/%d/%Y").strftime('%j')

                    # print(image.getexif().keys())
                    # for key, val in image.getexif().items():
                    #     if key in ExifTags.TAGS:
                    #         print(f'{ExifTags.TAGS[key]}: {val}, {key}, {ExifTags.TAGS[306]}')
                    # photo_time = 0
                    # break


                crop_label = CROP_LIST.index("Wheat")

                annotation_file.write(str(date_label) + " " + str(crop_label) + '\n')
                
                # for key, val in image.getexif().items():
                #     if key in ExifTags.TAGS:
                #         print(f'{ExifTags.TAGS[key]}: {val}, {key}')
                
                
                
                voc_annotation_file = open(os.path.join(root, f"{image_id}.xml"), encoding='utf-8')

                tree = ET.parse(voc_annotation_file)
                xml_root = tree.getroot()

                width = xml_root.find('size')[0].text
                height = xml_root.find('size')[1].text

                if int(width) == 0:
                    img = io.imread(os.path.join(root,file))
                    width = img.shape[1]
                    height = img.shape[0]

                for obj in xml_root.iter('object'):
                    cls = obj.find('name').text

                    # if cls == 'Other object' or cls == 'Other' or cls == 'other':
                    #     continue

                    if cls not in PEST_LIST:
                        PEST_LIST.append(cls)
                        cls_id = CLASSES_MAP[PEST_LIST.index(cls)]
                        classes_dict[cls_id] = 1
                    else:
                        cls_id = CLASSES_MAP[PEST_LIST.index(cls)]
                        classes_dict[cls_id] += 1

                    # cls_id = CLASSES_MAP[PEST_LIST.index(cls)]
                    # cls_id = 0
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                    bb = convert((width,height),b)
                    annotation_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

                annotation_file.write("\n")
                annotation_file.close()


    print(CROP_LIST, PEST_LIST, classes_dict)
    print(image_no_time)

def search_original_img(id_list, original_path):

    for folder_path in original_path:
        for root, folder, files in os.walk(folder_path):
            for file in files:
                image_id, file_type = os.path.splitext(file)
                if file_type in IMAGE_FILE_TYPE:
                    img = imread(os.path.join(root, file))
                    image_hash = aHash(img)
                    image_hash = hex(int(image_hash, 2))
                    if str(image_hash) in id_list:
                        print(f"{image_hash}: {os.path.join(root, file)}")

def renew_image(path, target_folder):
    for root, folders, files in os.walk(path):
        for file in files:
            image_id, file_type = os.path.splitext(file)
            if file_type in IMAGE_FILE_TYPE:
                img = imread(os.path.join(root, file))
                image_hash = aHash(img)
                image_hash = hex(int(image_hash, 2))
                print(image_hash)
                shutil.copyfile(os.path.join(root,file), os.path.join(target_folder ,f"{image_hash}.JPG"))


def stat_modal_info(path):
    DATA_FORMAT = '%Y:%m:%d %H:%M:%S'

    time = {}
    crops = {}
    for root, folder, files in os.walk(path):
        for file in files:
            image_id, file_type = os.path.splitext(file)
            if not (file_type == ".xml" or  file_type == ".txt"):
                image = Image.open(os.path.join(root, file))
                
                if 306 in image.getexif():
                    photo_time = image.getexif()[306]
                    date_label = int(datetime.strptime(photo_time, DATA_FORMAT).strftime('%j'))
                else:
                    # image_no_time.append(image_id)
                    date_label = 0

                print(date_label)

                if date_label in time:
                    time[date_label] += 1
                else:
                    time[date_label] = 1

    #        1, 2, 3, 4, 5,  6,  7,  8,  9,  10, 11,12
    month = [0,31,59,90,120,151,181,212,243,273,304,334]

    print(f"------------time------ \n{time}") 
    print(f"------------crop------ \n{crops}") 


if __name__ == "__main__":
    original_path = ["F:\\pest_data\\original_image"]
    target_path = "F:\\pest_data\\Multitask_or_multimodality\\annotated_data"

    # create_multitask_or_multimodel_dataset_from_xml(original_path, target_path)

    # path = "F:\\pest_data\\original_image\\Builted_Dataset_In_2023_Before_June_05\\IMG_8997.JPG"
    # test_img(path)

    # format_multimodal_annotation("F:\\pest_data\\Multitask_or_multimodality\\annotated_images")

    l = ['0x130b0df4faa39020', '0x27bf9f47172f8021', '0x38b8783e26078108', '0x3a19007fbc62067c', '0x3f4c4da9e50d1d9c', '0x50c02cecf0323c3', '0x61cafbaadfb8821c', '0x670626a633b8f9fc', '0x74b0c0e734d3bbfd', '0x7871d30e894c230d', '0x7eff5f0f07030118', '0x7f7658607064a073', '0x830203001d3e7f63', '0x9c99c0c48aeae481', '0x9e2be70e5c680a44', '0x9fc07a181fc3f0c4', '0xafce1ee0000e0d0', '0xbd5e1f0f070301e9', '0xc44dee3387c3bd86', '0xcbd648a79de8c0f7', '0xd0eb6484126811a5', '0xd73b1b23f920cd37', '0xe0aa743c390f1e48', '0xe0f8e1d838fef8', '0xe1fbb40280d880ec', '0xe20444fe105e8472', '0xe4f4f990e0f8f401', '0xf1b25b3d71108e87', '0xf4f8fce402110800', '0xf8fc9c80f00d32fc', '0xf8fcfcecfcfcf8f0', '0xfce4c1e771e0f86d', '0xfffff7e1c0c0e0e0', '0xfffffefafbfbd8ec']

    # for image_id in l:
    #     image = Image.open(os.path.join(target_path, "annotated_images",f"{image_id}.JPG"))
    #     print(image_id, image.getexif())
    # search_original_img(l, ["F:\\pest_data\\a"])
    # renew_image("F:\\pest_data\\b", "F:\\pest_data\\Multitask_or_multimodality\\annotated_images")

    stat_modal_info("F:\\pest_data\\Multitask_or_multimodality\\annotated_data")




