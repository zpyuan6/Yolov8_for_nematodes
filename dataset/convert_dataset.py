import os
import shutil
import xml.etree.ElementTree as ET
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from build_annotation import SetAnnotation
import numpy as np
from random import sample

CLASSES = ['pest', 'unknown', 'spider', 'fly', 'snail', 'aphid', 'slug', 'beetle', 'Pest', 'Fly', 'Beetle', 'snails', 'cabbage\\taphid']
# CLASSES_MAP = [0,0,1,2,3,4,5,6,0,2,6,3,4]
CLASSES_MAP = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,16,20,21,22,23,24,25,26,27,28,29,30]

def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 

def convert(size,box):

    x = (box[0] + box[2])/2.0 - 1
    y = (box[1] + box[3])/2.0 - 1
    x = x/float(size[0])
    y = y/float(size[1])

    # w = (box[3] - box[1])/float(size[0])
    # h = (box[2] - box[0])/float(size[1])
    w = (box[2] - box[0])/float(size[0])
    h = (box[3] - box[1])/float(size[1])

    return x,y,w,h

def convert_annotation(annotation_file_path, list_file, classes_list:list, classes_dict:dict, img_file_path=None, classes_name_list=None):
    voc_annotation_file = open(annotation_file_path, encoding='utf-8')
    tree = ET.parse(voc_annotation_file)
    root = tree.getroot()

    width = root.find('size')[0].text
    height = root.find('size')[1].text

    have_class_name_list = False if len(classes_list)==0 else True

    if int(width) == 0:
        img = io.imread(img_file_path)
        width = img.shape[1]
        height = img.shape[0]
        print(img.shape)

    for obj in root.iter('object'):
        class_name = obj.find('name').text.upper()

        # -----------All insecta start--------------
        # if 'APHID' in class_name:
        #     class_name = "GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID"

        # if 'CHIRONOMID MIDGE' in class_name:
        #     class_name = 'CHIRONOMID MIDGE'

        # if class_name in ['FROGHOPPER (CERCOPIDAE)', 'SCARABAEIDAE']:
        #     class_name = 'INSECTA'

        # if class_name in ['FLY (MELANOSTOMA SPP.)', 'BIBIONID FLY (BIBIONIDAE)', 'MUSCIDAE (FLY)', 'LEAF MINERS', 'BEAN SEED FLY (DELIA SPP.)', 'FUNGUS GNAT (MYCETOPHILIDAE)']:
        #     class_name = "FLY (DIPTERA)"
        # -----------All insecta end----------------

        # -----------Pest only start--------------
        if 'APHID' in class_name:
            class_name = "GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID"

        if not class_name in classes_list:
            class_name = "INSECTA (NOT CONCERNED)"
        # -----------Pest only end----------------
                    
        if not class_name in classes_list:
            print(class_name)
            # cls = "INSECTA (NOT CONCERNED)"
            # print(cls)
            class_name = "INSECTA"

        if have_class_name_list:
            # print(cls, " ", annotation_file_path)
            cls_id = classes_list.index(class_name)
            if cls_id in classes_dict:
                classes_dict[cls_id] += 1
            else:
                classes_dict[cls_id] = 1
        else:
            if class_name not in classes_list:
                classes_list.append(class_name)
                cls_id = CLASSES_MAP[classes_list.index(cls)]
                classes_dict[cls_id] = 1
            else:
                cls_id = CLASSES_MAP[classes_list.index(cls)]
                classes_dict[cls_id] += 1
            # cls_id = CLASSES_MAP[classes_list.index(cls)]

        # cls_id = 0
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


def convert_xml_to_yolo(source_folder, target_folder, classes_name_list=None):
    train_val = 0.8

    classes_dict = {}
    classes_list = [] if classes_name_list==None else classes_name_list
    sample_list_ordered_by_class = {}

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
            if not (file_type == ".xml" or file_type == ".txt"):
                voc_annotation_file = open(os.path.join(root,file_name+".xml"), encoding='utf-8')
                tree = ET.parse(voc_annotation_file)
                r = tree.getroot()
                temp_list = set([])
                for obj in r.iter('object'):
                    class_name = obj.find('name').text
                    # -----------All insecta start--------------
                    # if 'APHID' in class_name:
                    #     class_name = "GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID"

                    # if 'CHIRONOMID MIDGE' in class_name:
                    #     class_name = 'CHIRONOMID MIDGE'

                    # if class_name in ['FROGHOPPER (CERCOPIDAE)', 'SCARABAEIDAE']:
                    #     class_name = 'INSECTA'

                    # if class_name in ['FLY (MELANOSTOMA SPP.)', 'BIBIONID FLY (BIBIONIDAE)', 'MUSCIDAE (FLY)', 'LEAF MINERS', 'BEAN SEED FLY (DELIA SPP.)', 'FUNGUS GNAT (MYCETOPHILIDAE)']:
                    #     class_name = "FLY (DIPTERA)"
                    # -----------All insecta end----------------

                    # -----------Pest only start--------------
                    if 'APHID' in class_name:
                        class_name = "GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID"

                    if not class_name in classes_list:
                        class_name = "INSECTA (NOT CONCERNED)"
                    # -----------Pest only end----------------
                    
                    # if not class_name in classes_list:
                    #     class_name = "INSECTA (NOT CONCERNED)"
                        # print(class_name)
                        # class_name = "INSECTA"

                    temp_list.add(class_name)
                for c in temp_list:
                    if c in sample_list_ordered_by_class:
                        sample_list_ordered_by_class[c].append(file)
                    else:
                        sample_list_ordered_by_class[c] = [file]

    sample_list_ordered_by_class = dict(sorted(sample_list_ordered_by_class.items(), key=lambda item: len(item[1])))
    val_image_list = set([])
    train_image_list = set([])
    for c in sample_list_ordered_by_class.keys():
        # print(len(sample_list_ordered_by_class[c]))
        val_list = sample(sample_list_ordered_by_class[c], int(max(1, len(sample_list_ordered_by_class[c])*(1-train_val))))
        train_list = set(sample_list_ordered_by_class[c])-set(val_list)
        val_image_list = val_image_list.union(val_list)
        train_image_list = train_image_list.union(train_list)
        # print(len(val_list),len(train_list),len(val_image_list),len(train_image_list))


    for root, folders, files in os.walk(source_folder):
        for file in files:
            file_name,file_type = os.path.splitext(file)
            if not (file_type == ".xml" or file_type == ".txt"):
                if file in train_image_list:
                    shutil.copy(os.path.join(root,file),os.path.join(train_yolo_image_folder,file))

                    annotation_file = open(os.path.join(train_yolo_label_folder,file_name+".txt"), 'w', encoding='utf-8')
                    
                    convert_annotation(os.path.join(root,file_name+".xml"), annotation_file, classes_list, classes_dict, os.path.join(root,file))

                    annotation_file.write('\n')

                    annotation_file.close()
                else:
                    shutil.copy(os.path.join(root,file),os.path.join(val_yolo_image_folder,file))

                    annotation_file = open(os.path.join(val_yolo_label_folder,file_name+".txt"), 'w', encoding='utf-8')

                    convert_annotation(os.path.join(root,file_name+".xml"), annotation_file, classes_list, classes_dict, os.path.join(root,file))

                    annotation_file.write('\n')

                    annotation_file.close()


    #             ro = random.random()
    #             if ro < train_val:
    #                 # if file_type == ".JPG":
    #                 #     shutil.copy(os.path.join(root,file),os.path.join(train_yolo_image_folder,file_name+".jpg"))
    #                 # else:
    #                 shutil.copy(os.path.join(root,file),os.path.join(train_yolo_image_folder,file))

    #                 annotation_file = open(os.path.join(train_yolo_label_folder,file_name+".txt"), 'w', encoding='utf-8')

    #                 convert_annotation(os.path.join(root,file_name+".xml"), annotation_file, classes_list, classes_dict, os.path.join(root,file))

    #                 annotation_file.write('\n')

    #                 annotation_file.close()
    #             else:
    #                 # if file_type == ".JPG":
    #                 #     shutil.copy(os.path.join(root,file),os.path.join(val_yolo_image_folder,file_name+".jpg"))
    #                 # else:
    #                 shutil.copy(os.path.join(root,file),os.path.join(val_yolo_image_folder,file))

    #                 annotation_file = open(os.path.join(val_yolo_label_folder,file_name+".txt"), 'w', encoding='utf-8')

    #                 convert_annotation(os.path.join(root,file_name+".xml"), annotation_file, classes_list, classes_dict, os.path.join(root,file))

    #                 annotation_file.write('\n')

    #                 annotation_file.close()

    print(classes_list, classes_dict)
    

def convert_yolo_to_xml(source_folder, out_folder):
    print("Start convert")

    for root, folders, files in os.walk(source_folder):
        for file in files:
            image_path = os.path.join(root,file).replace("labels","images").replace("txt","JPG")
            # width, height
            img_shape = Image.open(image_path).size
            annotation = SetAnnotation("F:\\pest_data\\Multitask_or_multimodality\\annotated_data\\0x0.xml", root, out_folder, [
                "INSECTA",
                "GRAIN APHID (SITOBION AVENAE)",
                "POLLEN BEETLE (MELIGETHES SPP.)",
                "SNAIL",
                "POLYGONUM LEAF BETTLE (GASTROPHYSA POLYGONI)",
                "FLY",
                "CABBAGE STEM FLEA BETTLE",
                "COCCINELLIDAE (LADYBUG)",
                "SPIDER",
                "CHIRONOMID MIDGE",
                "COLEOPTERA",
                "MOSQUITO",
                "WASP",
                "SLUG",
                "FROG HOPPER",
                "ARANEUS SPP",
                "HEMIPTERA (PLANT BUG)",
                "EARTHWORM",
                "LEAF MINERS"
            ])
            bboxs = np.loadtxt(os.path.join(root,file))
            if bboxs.shape == (0,):
                continue
            if len(bboxs.shape)==1:
                bboxs = np.expand_dims(bboxs, 0)
            bboxs = np.insert(bboxs, 5, values=bboxs[:,0], axis=1)
            bboxs = np.insert(bboxs, 6, values=bboxs[:,0], axis=1)
            bboxs = np.delete(bboxs, 0, axis=1)
            bboxs[:,0] = bboxs[:,0]-(bboxs[:,2]/2)   
            bboxs[:,2] = bboxs[:,0]+bboxs[:,2]
            bboxs[:,1] = bboxs[:,1]-(bboxs[:,3]/2)   
            bboxs[:,3] = bboxs[:,1]+bboxs[:,3]
            annotation(file.split('.')[0], img_shape, bboxs)


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



def present_annotation(image_path, annotation_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    # image = plt.imread(image_path)
    # image = Image.open(image_path)
    # print(image_path)
    # image = io.imread(image_path)
    # print(type(image))


    def read_list(txt_path):
        pos = []
        with open(txt_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                    pass
                
                p_tmp = []
                for i in lines.split(' '):
                    if i!='\n':
                        p_tmp.append(float(i))

                if len(p_tmp)>0:
                    pos.append(p_tmp)

                pass
        return pos

    def unconvert(size, box):
        xmin = (box[1]-box[3]/2.)*size[1]
        xmax = (box[1]+box[3]/2.)*size[1]
        ymin = (box[2]-box[4]/2.)*size[0]
        ymax = (box[2]+box[4]/2.)*size[0]
        box = (int(xmin),int(ymin),int(xmax),int(ymax))
        return box
    
    pos = read_list(annotation_path)
    # tl = int((image.shape[0]+image.shape[1])/2) + 1
    # lf = max(tl-1,1)
    for i in range(len(pos)):
        print(i, pos)
        box = unconvert(image.shape, pos[i])
        print(box)
        image = cv2.rectangle(image, (box[0],box[1]),(box[2],box[3]), (255,0,0),10)
        pass

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.imshow(image)
    plt.show()


def check_annotation(yolo_folder):
    image_folder = os.path.join(yolo_folder,'images')
    annotation_folder = os.path.join(yolo_folder, 'labels')

    for root, folders, files in os.walk(image_folder):
        for file in files:
            folder = root.split("\\")[-1]
            print(os.path.join(root,file))
            present_annotation(os.path.join(root, file), os.path.join(annotation_folder,folder,file.split(".")[0]+".txt"))


def rename_file(folder, predix):

    for root, folders, files in os.walk(folder):

        for file in files:
            os.rename(os.path.join(root, file), os.path.join(root, f"{predix}_{file}"))

def copy_from_yolo(orign_yolo_folder,img_anno_folder,target_folder):

    id_list = []
    for root, folders, files in os.walk(os.path.join(orign_yolo_folder,"labels")):
        for file in files:
            if file.split(".")[-1] == "txt":
                img_id = file.split(".")[0]
                if img_id not in id_list:
                    id_list.append(img_id)
                    shutil.copy2(os.path.join(img_anno_folder, f"{img_id}.JPG"),os.path.join(target_folder, f"{img_id}.JPG"))
                    shutil.copy2(os.path.join(img_anno_folder, f"{img_id}.xml"),os.path.join(target_folder, f"{img_id}.xml"))

if __name__ == "__main__":
    org_path = "F:\\pest_data\\Multitask_or_multimodality\\annotated_data"
    # org_path = "F:\\pest_data\\Multitask_or_multimodality\\temp"
    # yolo_path = "F:\\pest_data\\Multitask_or_multimodality\\YOLO_18SEP24_ALL_INSECTA"
    yolo_path = "F:\\pest_data\\Multitask_or_multimodality\\YOLO_18SEP24_PEST_ONLY"
    # yolo_path = "F:\\pest_data\\Multitask_or_multimodality\\YOLO_18SEP24_ONLY_PEST_NO_FLY"

    # org_path = "F:\\nematoda\\AgriNema\\original_annotated_data"
    # yolo_path = "F:\\nematoda\\AgriNema\\Formated_Dataset\\Yolo_11Dec"
    # voc_path = "F:\\nematoda\\Microorganism\\Dataset"
    # yolo_path = "F:\\nematoda\\Microorganism\\YOLO"
    # copy_annotation(yolo_path, voc_path)
    # convert_xml_to_yolo(org_path, yolo_path)

    classes_name_list_for_only_pest = [
        "INSECTA (NOT CONCERNED)",
        "GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID",
        "POLLEN BEETLE (MELIGETHES SPP.)",
        "SNAIL",
        "CEREAL LEAF BEETLE (OULEMA MELANOPUS)",
        "CABBAGE STEM FLEA BETTLE",
        "SLUG",
        "CABBAGE WHITEFLY",
        "LEAF MINERS",
        "BEAN SEED FLY (DELIA SPP.)"
    ]

    classes_name_list_for_only_pest_no_fly = [
        "INSECTA (NOT CONCERNED)",
        "GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID",
        "POLLEN BEETLE (MELIGETHES SPP.)",
        "SNAIL",
        "CEREAL LEAF BEETLE (OULEMA MELANOPUS)",
        "CABBAGE STEM FLEA BETTLE",
        "SLUG",
        "CABBAGE WHITEFLY"
    ]

    classes_name_list_for_all_insecta = [
        "INSECTA", # 'INSECTA': 834, 'FROGHOPPER (CERCOPIDAE)': 1, 'SCARABAEIDAE': 1
        "GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID", # 'GRAIN APHID (SITOBION AVENAE)', 'ROSE GRAIN APHID': 130+1, 
        "POLLEN BEETLE (MELIGETHES SPP.)", # 'POLLEN BEETLE (MELIGETHES SPP.)': 2843, 
        "SNAIL", # 'SNAIL': 50, 
        "CEREAL LEAF BEETLE (OULEMA MELANOPUS)", # 'CEREAL LEAF BEETLE (OULEMA MELANOPUS)': 66, 
        "FLY (DIPTERA)", # 'FLY (DIPTERA)': 1210, 'FLY (MELANOSTOMA SPP.)': 14, 'BIBIONID FLY (BIBIONIDAE)': 41, 'MUSCIDAE (FLY)': 4, 'LEAF MINERS': 2, 'BEAN SEED FLY (DELIA SPP.)': 49, 'FUNGUS GNAT (MYCETOPHILIDAE)': 8, 
        "CABBAGE STEM FLEA BETTLE", #'CABBAGE STEM FLEA BETTLE': 123, 
        "LADYBUG (COCCINELLIDAE)", #'LADYBUG (COCCINELLIDAE)': 198, 
        "LADYBUG (COCCINELLIDAE) (PUPA)", #'LADYBUG (COCCINELLIDAE) (PUPA)': 166, 
        "LADYBUG (COCCINELLIDAE) (LARVAE)", #'LADYBUG (COCCINELLIDAE) (LARVAE)': 10, 
        "SPIDER (ARANEUS SPP.)", #'SPIDER (ARANEUS SPP.)': 73, 
        "CHIRONOMID MIDGE", #'CHIRONOMID MIDGE': 93, 'CHIRONOMID MIDGE (MALE)': 4,
        "BEETLE (COLEOPTERA)", #'BEETLE (COLEOPTERA)': 176, 
        "MOSQUITO", # 'MOSQUITO': 77,
        "WASP", #'WASP': 36, 
        "SLUG", #'SLUG': 67, 
        "CABBAGE WHITEFLY", #'CABBAGE WHITEFLY': 13,
        "BEE", # 'BEE': 35,  
        "HEMIPTERA (PLANT BUG)", #'HEMIPTERA (PLANT BUG)': 4,
        "EARTHWORM", #'EARTHWORM': 7, 
        "BUMBLEBEE", # 'BUMBLEBEE': 70, 
        "GROUND BETTLE (HARPALUS SPP)", #'GROUND BETTLE (HARPALUS SPP)': 18, 
        "ANT", #'ANT': 10, 
        "PYRRHOCORIDAE", #'PYRRHOCORIDAE': 7, 
        "LONGICORN" #'LONGICORN': 5
    ]

    convert_xml_to_yolo(org_path, yolo_path, classes_name_list_for_only_pest_no_fly)

    # copy_from_yolo("F:\\pest_data\Multitask_or_multimodality\\YOLO_24Dec", "F:\\pest_data\\Multitask_or_multimodality\\annotated_data", "F:\\pest_data\\Multitask_or_multimodality\\temp")

    # convert_yolo_to_xml("F:\\pest_data\\Multitask_or_multimodality\\YOLO_01JAN\\labels","F:\\pest_data\\Multitask_or_multimodality\\VOCdevkit\\VOC2007\\Annotations")

    # check_annotation(yolo_path)
    # # print(os.path.exists("F:\\Pest\\pest_data\\yolo\\images\\train\\IMG_7544.JPG"))
    # present_annotation("F:\\Pest\\pest_data\\yolo\\images\\train\\IMG_8059.JPG", "F:\\Pest\\pest_data\\yolo\\labels\\train\\IMG_8059.txt")
    # train_list = ["IMG_7419","IMG_7422","IMG_7423","IMG_7424","IMG_7425","IMG_7426","IMG_7427","IMG_7428","IMG_7429","IMG_7430","IMG_7548","IMG_7552","IMG_7558","IMG_7559","IMG_7572","IMG_7574","IMG_7575","IMG_7579","IMG_7580","IMG_7587","IMG_7597","IMG_7598","IMG_7599","IMG_7600","IMG_7601","IMG_7602","IMG_7603","IMG_7604","IMG_7605","IMG_7617","IMG_7618","IMG_7619","IMG_7620","IMG_762","IMG_7622","IMG_7623","IMG_7987","IMG_7988","IMG_7991","IMG_7994","IMG_7995","IMG_7996","IMG_7997","IMG_7998","IMG_7999","IMG_8001","IMG_8003","IMG_8004","IMG_8005","IMG_8006","IMG_8007","IMG_8009","IMG_8010","IMG_8013","IMG_8014","IMG_8015","IMG_8016","IMG_8017","IMG_8018","IMG_8023","IMG_8024","IMG_8025","IMG_8026","IMG_8027","IMG_8029","IMG_8030","IMG_8031","IMG_8033","IMG_8034","IMG_8034","IMG_8036","IMG_8037","IMG_8038","IMG_8039","IMG_8040","IMG_8041","IMG_8043","IMG_8045","IMG_8046","IMG_8047","IMG_8048","IMG_8049","IMG_8050","IMG_8051","IMG_8053","IMG_8054","IMG_8055","IMG_8056","IMG_8057","IMG_8058","IMG_8059","IMG_8060","IMG_8061","IMG_8062","IMG_8063","IMG_8064","IMG_8065","IMG_8066","IMG_8067","IMG_8068","IMG_8069","IMG_8071","IMG_8073","IMG_8075","IMG_8076","IMG_8077","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""]
    # val_list = []

    # rename_file("F:\\nematoda\\AgriNema\\unannotated_data\\PCN_RLN_JPEG_unfinished\\Original_annotation","pcn_rln_x5")