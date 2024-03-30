from ultralytics import YOLO

def export_ptl(model_path):
    model = YOLO(model_path)
    model.export(format='torchscript', dynamic=True, optimize=True)

# def run_ptl(model):

def export_ncnn(model_path):
    model = YOLO(model_path)
    model.export(format='onnx', imgsz=640, half=True, opset = 12)
    # model.export(format='ncnn', half=True, imgsz=640)

if __name__ == "__main__":
    # model_path = "runs\\detect\\our_medium_all\\weights\\best.pt"
    # export_ptl(model_path)

    # model_path = "runs\\uk_pest_01JAN_medium\\weights\\best.pt"

    model_path = "runs/detect/nematodes_tiny_31_083/weights/best.pt"

    export_ncnn(model_path)