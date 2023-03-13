from ultralytics import YOLO
import cv2


# Yolov8 DOCS https://docs.ultralytics.com/modes/predict/
if __name__ == "__main__":
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