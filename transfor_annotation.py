from PIL import Image
import os
import cv2
import numpy as np
import json
import tqdm 

def generate_bounding_box_json():
    mode = ["test", "train", "val"]

    for m in mode:
        img_path = f"F:\\nematoda\\dataset\\{m}\\img"
        mask_path = f"F:\\nematoda\\dataset\\{m}\\mask"

        img_file_id = []
        for root, folder, files in os.walk(img_path):
            img_file_id.extend(files)

        bounding_boxes = {}
        for root, folder, files in os.walk(mask_path):
            with tqdm.tqdm(total=len(files)) as tbar:
                for file in files:
                    image_id = "_".join(file.split("_")[1:])
                    # print("_".join(file.split("_")[1:]))

                    mask = cv2.imread(os.path.join(root,file))
                    x = np.where(np.max(mask,axis=0)==np.max(mask))[0]
                    x1,x2 = x[0], x[-1]
                    y = np.where(np.max(mask,axis=1)==np.max(mask))[0]
                    y1,y2 = y[0], y[-1]
                    # print(x1,x2,y1,y2)

                    if image_id in bounding_boxes:
                        bounding_boxes[image_id].append([int(x1),int(y1),int(x2),int(y2)])
                    else:
                        bounding_boxes[image_id] = [[int(x1),int(y1),int(x2),int(y2)]]

                    tbar.update(1)
                    # image = cv2.imread("F:\\nematoda\\dataset\\test\\img\\img_352_id1.jpg")
                    # draw_1 = cv2.rectangle(image, (x1,y1), (x2,y2),(0,255,0),2)
                    # cv2.imshow("result",draw_1)
                    # cv2.waitKey(0)
        j = json.dumps(bounding_boxes)
        f = open(f"F:\\nematoda\\dataset\\{m}\\bounding_boxes.json","w")
        f.write(j)
        f.close()  

def generate_annotation_txt():
    mode = ["test", "train", "val"]

    for m in mode:
        bounding_boxes_file_path = f"F:\\nematoda\\dataset\\{m}\\bounding_boxes.json"
        f = open(bounding_boxes_file_path,"r")
        bounding_boxes = json.load(f)
        f.close()
        if not os.path.exists(f"F:\\nematoda\\dataset\\{m}\\labels"):
            os.makedirs(f"F:\\nematoda\\dataset\\{m}\\labels")

        for k in bounding_boxes:
            image_id = k.split(".")[0]
            annotation_file = open(f"F:\\nematoda\\dataset\\{m}\\labels\\{image_id}.txt","w")

            writing_content = []
            for bounding_box in bounding_boxes[k]:
                # 3280 2464
                # class x_center y_center width height https://blog.paperspace.com/train-yolov5-custom-data/
                writing_content.append(f"0 {(bounding_box[0]+bounding_box[2])/2/3280} {(bounding_box[1]+bounding_box[3])/2/2464} {(bounding_box[2]-bounding_box[0])/3280} {(bounding_box[3]-bounding_box[1])/2464}")

            annotation_file.writelines(writing_content)
            annotation_file.close()


if __name__ == "__main__":
    generate_bounding_box_json()

    generate_annotation_txt()

                

