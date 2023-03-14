from ultralytics import YOLO
import cv2


def prediction_test():
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    results = model("bus.jpg")  # predict on an image
    # success = model.export(format="onnx")  # export the model to ONNX format

    res_plotted = results[0].plot()

    print("Results class", results[0].probs)
    #  format x1,y1,x2,y2,conf,cls
    print("Bounding box", results[0].boxes)
    print("Bounding box", results[0].boxes.xyxy)

    print(results[0].masks)
    print(results[0].probs)

    print(type(res_plotted))
    cv2.imshow("result", res_plotted)
    cv2.waitKey()
    # print(results[0].boxes)

def prediction(model_path,image_path):
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO(model_path)  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    results = model(image_path)  # predict on an image

    # print("Results class", results[0].probs)
    # #  format x1,y1,x2,y2,conf,cls
    # print("Bounding box", results[0].boxes)
    # print("Bounding box", results[0].boxes.xyxy)

    img = cv2.imread(image_path)
    cv2.namedWindow("result",0)

    for bounding_box in results[0].boxes.xyxy.cpu().numpy():
        print(bounding_box)
        print(bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3])
        img = cv2.rectangle(img, (int(bounding_box[0]),int(bounding_box[1])), (int(bounding_box[2]),int(bounding_box[3])),(0,255,0),2)

    cv2.imshow("result", img)
    cv2.waitKey()
    # print(results[0].boxes)


# Yolov8 DOCS https://docs.ultralytics.com/modes/predict/
if __name__ == "__main__":
    prediction("runs\\detect\\train\\weights\\last.pt", "F:\\nematoda\\nemadote_detection\\images\\train\\img_038_id0.jpg")