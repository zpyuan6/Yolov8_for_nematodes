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
                    if obj.find('name').text == 'LADYBUG (COCCINELLIDAE) (LARVEA)':
                        obj.find('name').text = 'LADYBUG (COCCINELLIDAE) (LARVAE)'.upper()

                    if obj.find('name').text == 'GROUND BETTLE(HARPALUS SPP)':
                        obj.find('name').text = 'GROUND BETTLE (HARPALUS SPP)'.upper()

                    if obj.find('name').text == 'SPIDER':
                        obj.find('name').text = 'SPIDER (ARANEUS SPP.)'.upper()

                    # if obj.find('name').text == 'FLY':
                    #     obj.find('name').text = 'FLY (DIPTERA)'.upper()

                    # if obj.find('name').text == 'POLLEN BEETLE':
                    #     obj.find('name').text = 'POLLEN BEETLE (MELIGETHES SPP.)'.upper()

                    # if obj.find('name').text == 'POLLEN BEETLE':
                    #     obj.find('name').text = 'POLLEN BEETLE (MELIGETHES SPP.)'.upper()

                        # print(obj.find('name').text)
                    if not obj.find('name').text in name_list:
                        name_list.append(obj.find('name').text)

                    bnd = obj.find('bndbox')

                    bnd.find('xmin').text = f"{int(float(bnd.find('xmin').text)) }"
                    bnd.find('ymin').text = f"{int(float(bnd.find('ymin').text))}"
                    bnd.find('xmax').text = f"{int(float(bnd.find('xmax').text))}"
                    bnd.find('ymax').text = f"{int(float(bnd.find('ymax').text))}"

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


def move_manual_checked_data(source_path, target_path):

    for r, folders, files in os.walk(source_path):
        for file in files:
            if file.split(".")[-1] == "JPG":
                new_file_name = "-".join([r.split("\\")[-1], file.split(".")[0]])
                shutil.copy2(os.path.join(r, file), os.path.join(target_path, new_file_name+".JPG"))
                shutil.copy2(os.path.join(r, ".".join(file.split(".")[0:-1])+".xml"), os.path.join(target_path, new_file_name+".xml"))


if __name__=="__main__":
    path = "F:\\pest_data\\Multitask_or_multimodality\\annotated_data"
    # list_object_name(path)
    rename_object(path)
    # ['LADYBUG (COCCINELLIDAE)', 'POLLEN BEETLE (MELIGETHES SPP.)', 'BEETLE (COLEOPTERA)', 'INSECTA', 'FLY (DIPTERA)', 'BEETLE', 'FLY', 'GROUND BETTLE (HARPALUS SPP)', 'SPIDER', 'LADYBUG', 'APHID', 'CEREAL LEAF BEETLE (OULEMA MELANOPUS)']
    

    # move_one_classes_file(path, "F:\\pest_data\\Multitask_or_multimodality\\temp_data", ['INSECTA'])

    # move_manual_checked_data(path, "F:\\pest_data\\Multitask_or_multimodality\\annotated_video")
