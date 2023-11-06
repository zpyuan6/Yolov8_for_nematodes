import os

def image_number(path):
    num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            num+=1

    print(num)
            
def count_object_from_yolo_label_folder(path):
    num = 0
    object_stat = {}
    file_stat = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "txt":
                f = open(os.path.join(root,file),"r")
                lines = f.readlines()
                num += len(lines)
                if len(lines[0])>1:
                    class_id = int(lines[0].split(" ")[0])
                    if class_id in file_stat:
                        file_stat[class_id] += 1
                    else:
                        file_stat[class_id] = 1

                for line in lines:
                    if len(line)>1:
                        class_id = int(line.split(" ")[0])
                        if class_id in object_stat:
                            object_stat[class_id] += 1
                        else:
                            object_stat[class_id] = 1
    print(num)
    print(object_stat)
    print(file_stat)

def calculate_object_area(path):
    num = 0
    object_area = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "txt":
                f = open(os.path.join(root,file),"r")
                lines = f.readlines()
                num += len(lines)
                for line in lines:
                    arr = line.split(" ")
                    w,h  = arr[3],arr[4]
                    object_area += float(w)*float(h)

    print(object_area)   
    print(num)
    print(object_area/num)   

if __name__ == "__main__":
    # image_number("F:\\nematoda\\nema")
    count_object_from_yolo_label_folder("F:\\nematoda\\AgriNema\\Formated_Dataset\\Yolo_0831\\labels")
    # calculate_object_area("F:\\nematoda\\nemadote_detection\\labels")