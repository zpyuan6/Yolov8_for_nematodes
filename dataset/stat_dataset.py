import os

def image_number(path):
    num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            num+=1

    print(num)
            
def count_object(path):
    num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "txt":
                f = open(os.path.join(root,file),"r")
                lines = f.readlines()
                num += len(lines)
    print(num)

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
    # count_object("F:\\nematoda\\nemadote_detection\\labels")
    calculate_object_area("F:\\nematoda\\nemadote_detection\\labels")