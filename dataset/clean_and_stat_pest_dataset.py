import os
import shutil

def stat_images_and_annotation(root_folder):
    file_stat = {}
    xml_list = []
    for root, folders, files in os.walk(root_folder):
        for file in files:
            file_name,file_type = os.path.splitext(file)

            if file_type in file_stat:
                file_stat[file_type] += 1
            else:
                file_stat[file_type] = 1

            if file_type == ".xml":
                if file_name in xml_list:
                    print(os.path.join(root,file))
                else:
                    xml_list.append(file_name)
    
    print(file_stat)

IMAGE_TYPE = ['.jpg', '.JPG', '.JPEG', '.PNG', '.heic']

def transfer_file_to_build_dataset(root_folder, target_folder):
    def copy_file(src_path, obj_path, file_name, file_type):
        if os.path.exists(os.path.join(target_folder,file_name+file_type)):
            tag = root.split("\\")[-1]
            shutil.copy(os.path.join(root,file), os.path.join(target_folder,file_name+f"_{tag}"+file_type))
        else:
            shutil.copy(os.path.join(root,file), os.path.join(target_folder,file_name+file_type))


    for root, folders, files in os.walk(root_folder):
        for file in files:
            file_name,file_type = os.path.splitext(file)
            # get annotation file
            if file_type == ".xml":

                copy_file(root, target_folder, file_name,file_type)

                not_finded_file = True
                for optional_img_type in IMAGE_TYPE:
                    if file_name+optional_img_type in files:
                        copy_file(root, target_folder, file_name,optional_img_type)
                        not_finded_file = False
                        break
                
                if not_finded_file:
                    root_list = root.split("\\")
                    if "VOC2007" in root_list:
                        print("Please copy file for annotation: ", os.path.join(root, file))
                    else:
                        print("can not find img file for annotation: ", os.path.join(root, file))


if __name__ == "__main__":
    root_folder = "F:\\Pest\\pest_data\\UK_Pest_Original"
    stat_images_and_annotation(root_folder)

    target_folder = "F:\Pest\pest_data\Pest_Dataset_2023"
    transfer_file_to_build_dataset(root_folder, target_folder)
    stat_images_and_annotation(target_folder)
