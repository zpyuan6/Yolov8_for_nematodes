import os

if __name__ == "__main__":
    path = "F:\\pest_data\\Multitask_or_multimodality\\YOLO_01JAN\\labels\\train"
    target_id = 7
    for root,  folders, files in os.walk(path):
        for file in files:
            # print("---------------------------------")
            f = open(os.path.join(root,file),"r")
            lines = f.readlines()
            
            for line in lines:
                if len(line)>1:
                    class_id = int(line.split(" ")[0])
                    if class_id == target_id:
                        print(file.split(".")[0])
                        break