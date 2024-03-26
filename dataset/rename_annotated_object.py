import os
import shutil
import xml.etree.ElementTree as ET

def rename_object(path):
    name_list = []

    for r, folders, files in os.walk(path):
        for file in files:
            if file.split(".")[-1]=="xml":
                annotation_file = os.path.join(r, file)
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                objects = root.findall('object')

                for obj in objects:
                    if obj.find('name').text == 'FROGHOPPERS (CERCOPIDAE)':
                        obj.find('name').text = 'FROGHOPPER (CERCOPIDAE)'.upper()

                    if obj.find('name').text == 'LADYBUG (COCCINELLIDAE) (LARVEA)':
                        obj.find('name').text = 'LADYBUG (COCCINELLIDAE) (LARVAE)'.upper()

                        # print(obj.find('name').text)
                    if not obj.find('name').text in name_list:
                        name_list.append(obj.find('name').text)

                tree.write(annotation_file)
    
    print(name_list, len(name_list))

def list_object_name(path):
    name_list = []

    for r, folders, files in os.walk(path):
        for file in files:
            if file.split(".")[-1]=="xml":
                annotation_file = os.path.join(r, file)
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                objects = root.findall('object')

                for obj in objects:

                    if not obj.find('name').text in name_list:
                        name_list.append(obj.find('name').text)

                tree.write(annotation_file)

    print(name_list, len(name_list))


def move_one_classes_file(path, target_folder, target_classes):

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    copy_file_id_set = set([])

    for r, folders, files in os.walk(path):
        for file in files:
            if file.split(".")[-1]=="xml":
                annotation_file = os.path.join(r, file)
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                objects = root.findall('object')

                for obj in objects:
                    if obj.find('name').text in target_classes:
                        copy_file_id_set.add(file.split(".")[0])
    
    for r, folders, files in os.walk(path):
        for file in files:
            if file.split(".")[0] in copy_file_id_set:
                shutil.copy2(os.path.join(r,file), os.path.join(target_folder,file))



if __name__=="__main__":
    path = "F:\\pest_data\\Multitask_or_multimodality\\annotated_data"
    rename_object(path)

    # list_object_name(path)

    # move_one_classes_file(path, "F:\\pest_data\\Multitask_or_multimodality\\temp_data", ['INSECTA'])