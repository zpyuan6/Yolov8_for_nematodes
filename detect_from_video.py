from ultralytics import YOLO
import os
import cv2

if __name__ == "__main__":
    folder = "F:\\pest_data\\unannotated\\2024\\Video"
    video_name = "VID_20240502_154618.mp4"
    save_path = "d_output.mp4"
    video_path = os.path.join(folder, video_name)

    model = YOLO("runs\\detect\\YOLOv8_1080_25JUN24_pest_only_no_fly_tiny\\weights\\best.pt")

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break
        results = model(source=frame, save=True)
        result = results[0]
        anno_frame = result.plot()
        out.write(anno_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Finished')
