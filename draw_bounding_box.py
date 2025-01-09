import cv2
import time

def draw_bounding_box(image_path, bounding_box):
    """
    Draw bounding box on the image from annotation
    :param image_path: path to the image
    :param bounding_box: bounding box annotation
    """
    img = cv2.imread(image_path)
    cv2.namedWindow("result",0)

    print(img.shape)

    for i, bounding_box in enumerate(bounding_box):
        # if i==0:
        #     color = (0,0,192)
        # elif i==1:
        #     color = (17,90,197)
        if i>4:
            color = (0,144,191)
        else:
            color = (53,129,83)

        # print(bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3])
        img = cv2.rectangle(img, (int(bounding_box[0]),int(bounding_box[1])), (int(bounding_box[2]),int(bounding_box[3])),color,20)


    # num = len(results[0].boxes.xyxy.cpu())
    # class_result = results[0].names[int(results[0].boxes.cpu().data[i][-1].item())]
    cv2.imshow("result", img)
    cv2.waitKey()


    cv2.imwrite(f"result.png",img)

if __name__ == "__main__":

    # bounding_box = [
    #     [202,49,438,352],
    #     [205,52,339,162],
    #     [1,1,276,208],
    #     [220,17,275,87]
    # ]

    # draw_bounding_box("X:\\pervasive_group\\Shared\\flickr30k\\images\\97406261.jpg",bounding_box)

    bounding_box = [
        [1640,1159,1930,1340],
        [1369,578,1733,668],
        [2129,41,2316,165],
        [2319,662,2566,971],

        [460,1775,711,1973],
        [802,1443,1066,1732],
        [2225,1035,2577,1195],
        [2570,1319,2688,1437]
    ]


    draw_bounding_box("F:\\nematoda\\AgriNema\\original_annotated_data\\pcn_rln_x5_Image060_ch00.jpg", bounding_box)
