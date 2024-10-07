from ultralytics import YOLO

def export_ptl(model_path):
    model = YOLO(model_path)
    model.fuse()
    model.info(verbose=True)
    model.export(format='torchscript', imgsz=640)

    # model.export(format='torchscript', dynamic=True, optimize=True)

# def run_ptl(model):

def export_ncnn(model_path):
    model = YOLO(model_path)
    model.export(format='onnx', imgsz=640, half=True, opset = 11)
    # model.export(format='onnx', imgsz=640, half=True)
    # model.export(format='onnx', imgsz=640)

if __name__ == "__main__":
    # model_path = "runs\\detect\\our_medium_all\\weights\\best.pt"
    # export_ptl(model_path)

    model_path = "D:\\pest_object\\Yolov8_for_nematodes\\runs\\detect\\YOLO_25JUN24_ALL_INSECT_medium2\\weights\\best.pt"

    export_ptl(model_path)