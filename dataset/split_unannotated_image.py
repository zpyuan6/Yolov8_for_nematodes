import os
import shutil


if __name__ == "__main__":
    path = "F:\\pest_data\\Multitask_or_multimodality\\unannotated_images"

    index = 0
    for root, folders, files in os.walk(path):
        if root!=path:
            target_path = os.path.join(root, str(index))
            os.makedirs(target_path)
            for f_id in file_list:
                shutil.copyfile(os.path.join(root, f_id+".JPG"),os.path.join(target_path,f_id+".JPG"))
                if os.path.join(root, f_id+".xml"):
                    shutil.copyfile(os.path.join(root, f_id+".xml"),os.path.join(target_path,f_id+".xml"))
            index +=1
            file_list = []
            break

        file_list = []
        for file in files:
            file_id = file.split(".")[0]
            file_type = file.split(".")[-1]
            if file_type == "JPG":
                file_list.append(file_id)
                
            if len(file_list)==200:
                target_path = os.path.join(root, str(index))
                os.makedirs(target_path)
                for f_id in file_list:
                    shutil.copyfile(os.path.join(root, f_id+".JPG"),os.path.join(target_path,f_id+".JPG"))
                    if os.path.exists(os.path.join(root, f_id+".xml")):
                        shutil.copyfile(os.path.join(root, f_id+".xml"),os.path.join(target_path,f_id+".xml"))
                
                index +=1
                file_list = []

                        

                    


            

