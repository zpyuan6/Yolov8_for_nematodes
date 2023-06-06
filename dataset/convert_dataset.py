import os
import shutil
import xml.etree.ElementTree as ET
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io

CLASSES = ['pest', 'unknown', 'spider', 'fly', 'snail', 'aphid', 'slug', 'beetle', 'Pest', 'Fly', 'Beetle', 'snails', 'cabbage\\taphid']
CLASSES_MAP = [0,0,1,2,3,4,5,6,0,2,6,3,4]
# CLASSES_MAP = [0,1,2,3,4,5,6]

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

def convert_annotation(annotation_file_path, list_file, classes_list:list, classes_dict:dict):
    voc_annotation_file = open(annotation_file_path, encoding='utf-8')
    tree = ET.parse(voc_annotation_file)
    root = tree.getroot()

    width = root.find('size')[0].text
    height = root.find('size')[1].text

    for obj in root.iter('object'):
        cls = obj.find('name').text

        print(cls)

        if cls == 'Other object':
            continue

        if cls not in classes_list:
            classes_list.append(cls)
            cls_id = CLASSES_MAP[classes_list.index(cls)]
            classes_dict[cls_id] = 1
        else:
            cls_id = CLASSES_MAP[classes_list.index(cls)]
            classes_dict[cls_id] += 1

        cls_id = CLASSES_MAP[classes_list.index(cls)]
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


def convert_xml_to_yolo(source_folder, target_folder):
    train_val = 0.8

    classes_dict = {}
    classes_list = []

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
                    # if file_type == ".JPG":
                    #     shutil.copy(os.path.join(root,file),os.path.join(train_yolo_image_folder,file_name+".jpg"))
                    # else:
                    shutil.copy(os.path.join(root,file),os.path.join(train_yolo_image_folder,file))

                    annotation_file = open(os.path.join(train_yolo_label_folder,file_name+".txt"), 'w', encoding='utf-8')

                    convert_annotation(os.path.join(root,file_name+".xml"),annotation_file, classes_list, classes_dict)

                    annotation_file.write('\n')

                    annotation_file.close()
                else:
                    # if file_type == ".JPG":
                    #     shutil.copy(os.path.join(root,file),os.path.join(val_yolo_image_folder,file_name+".jpg"))
                    # else:
                    shutil.copy(os.path.join(root,file),os.path.join(val_yolo_image_folder,file))

                    annotation_file = open(os.path.join(val_yolo_label_folder,file_name+".txt"), 'w', encoding='utf-8')

                    convert_annotation(os.path.join(root,file_name+".xml"),annotation_file, classes_list, classes_dict)

                    annotation_file.write('\n')

                    annotation_file.close()

    print(classes_list, classes_dict)
    

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


if __name__ == "__main__":
    # voc_path = "F:\\Pest\\pest_data\\Pest_Dataset_2023"
    # yolo_path = "F:\\Pest\\pest_data\\YOLO_All_Classes_2023"
    voc_path = "F:\\nematoda\\our_dataset\\original_mix_data"
    yolo_path = "F:\\nematoda\\our_dataset\\YoloDataset"
    # copy_annotation(yolo_path, voc_path)
    convert_xml_to_yolo(voc_path, yolo_path)

    # check_annotation(yolo_path)
    # # print(os.path.exists("F:\\Pest\\pest_data\\yolo\\images\\train\\IMG_7544.JPG"))
    # present_annotation("F:\\Pest\\pest_data\\yolo\\images\\train\\IMG_7881.JPG", "F:\\Pest\\pest_data\\yolo\\labels\\train\\IMG_7544.txt")
    # train_list = ['0_28','0_29',"0_31","0_32","0_35","1_2","1_3","1_4","2_122","2_123","2_124","2_125","2_126","2_127","2_128","2_131","2_132","2_68","2_69","2_70","2_71","2_72","2_77","2_78","2_83","2_84","2_85","2_86","2_87","2_89","3_20","3_21","3_22","3_23","3_24","3_25","3_26","3_27","3_29","3_30","3_39","3_41","3_44","3_45","3_46","3_47","3_48","3_49","3_50","3_51","3_53","3_54","3_55","3_56","3_57","3_58","3_59","3_60","3_61","3_63","3_65","3_67","3_68","3_69","4_6","4_7","5_17","5_19","5_20","6_10","IMG_20230323_151919","IMG_3300","IMG_3301","IMG_3302","IMG_3303","IMG_3304","IMG_7353","IMG_7354","IMG_7355","IMG_7356","IMG_7361","IMG_7362","IMG_7363","IMG_7366","IMG_7371","IMG_7373","IMG_7374","IMG_7376","IMG_7377","IMG_7378","IMG_7380","IMG_7381","IMG_7382","IMG_7383","IMG_7384","IMG_7385","IMG_7386","IMG_7388","IMG_7389","IMG_7394","IMG_7395","IMG_7396","IMG_7397","IMG_7403","IMG_7404","IMG_7405","IMG_7408","IMG_7409","IMG_7411","IMG_7412","IMG_7413","IMG_7414","IMG_7415","IMG_7416","IMG_7417","IMG_7419","IMG_7422","IMG_7423","IMG_7424","IMG_7425","IMG_7426","IMG_7427","IMG_7428","IMG_7429","IMG_7430","IMG_7548","IMG_7552","IMG_7558","IMG_7559","IMG_7572","IMG_7574","IMG_7575","IMG_7579","IMG_7580","IMG_7587","IMG_7597","IMG_7598","IMG_7599","IMG_7600","IMG_7601","IMG_7602","IMG_7603","IMG_7604","IMG_7605","IMG_7617","IMG_7618","IMG_7619","IMG_7620","IMG_762","IMG_7622","IMG_7623","IMG_7987","IMG_7988","IMG_7991","IMG_7994","IMG_7995","IMG_7996","IMG_7997","IMG_7998","IMG_7999","IMG_8001","IMG_8003","IMG_8004","IMG_8005","IMG_8006","IMG_8007","IMG_8009","IMG_8010","IMG_8013","IMG_8014","IMG_8015","IMG_8016","IMG_8017","IMG_8018","IMG_8023","IMG_8024","IMG_8025","IMG_8026","IMG_8027","IMG_8029","IMG_8030","IMG_8031","IMG_8033","IMG_8034","IMG_8034","IMG_8036","IMG_8037","IMG_8038","IMG_8039","IMG_8040","IMG_8041","IMG_8043","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""]
    # val_list = []