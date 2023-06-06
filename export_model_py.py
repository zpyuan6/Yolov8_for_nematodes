from ultralytics import YOLO

def export_ptl(model_path):
    model = YOLO(model_path)
    model.export(format='torchscript', dynamic=True, optimize=True)

# def run_ptl(model):


if __name__ == "__main__":
    # model_path = "runs\\detect\\our_medium_all\\weights\\best.pt"
    # export_ptl(model_path)

    model_path = ""