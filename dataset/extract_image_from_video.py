import cv2
import os
import numpy as np
import tqdm
import shutil
from skimage.metrics import structural_similarity

# Structural Similarity Index (SSI) is a metric that measures the similarity between two images.
def similarity_check(img1,img2, threthold = 0.5):
    grayA = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    (score, diff) = structural_similarity(grayA, grayB, full=True)

    print(score)

    if score<threthold:
        return True
    else:
        return False

def similarity_check_based_on_optical_flow(img1, img2, threshold=2):

    grayA = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(grayA, grayB, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_magnitude = np.mean(magnitude)

    # print(mean_magnitude)

    if mean_magnitude>threshold:
        return True
    else:
        return False



def rename_file_type(path):

    for root, folders, files in os.walk(path):
        for file in files:
            file_name = file.split(".")[0]
            file_type = file.split(".")[-1]
            if file_type=="jpeg":
                print(file)
                os.rename(os.path.join(root, file), os.path.join(root, f"{file_name}.JPG"))

def extract_img_from_video():

    # video_path = "F:\\pest_data\\unannotated\\Videos"
    video_path = "F:\\pest_data\\unannotated\\Annotated_Videos"
    image_path = "F:\\pest_data\\unannotated\\Annotated_Videos"

    for root, folders, files in os.walk(video_path):
        with tqdm.tqdm(total=len(files)) as tbar:
            for file in files: 
                if file.split(".")[-1] != "MOV":
                    continue

                if not os.path.exists(os.path.join(image_path,file.split(".")[0])):
                    os.mkdir(os.path.join(image_path,file.split(".")[0]))

                tbar.update(1)

                cam = cv2.VideoCapture(os.path.join(root,file))
                frameno=0
                img1 = 0
                while(1):
                    ret,frame = cam.read()

                    if ret:
                        name = 'frame_'+("%06d" % frameno)+".JPG"

                        if frameno == 0 or similarity_check_based_on_optical_flow(img1, frame):
                            img1 = frame
                            cv2.imwrite(os.path.join(image_path,file.split(".")[0],name), frame)
                            # cv2.imshow(name, frame)
                            # cv2.waitKey(0)
                        frameno +=1
                    else:
                        break

                cam.release()
            # cv2.destroyAllWindows()

def extract_img_from_img_folder():

    old_image_folder = "F:\\pest_data\\unannotated\\2024\\Image_From_Video"
    image_path = "F:\\pest_data\\unannotated\\2024\\Image_From_Video_0.3"

    for root, folders, files in os.walk(old_image_folder):
        with tqdm.tqdm(total=len(files)) as tbar:
            img1 = 0
            video_id = 0
            for file in files: 

                new_video_id = "_".join(file.split("_")[:-1]) 

                if video_id == new_video_id:
                    img2 = cv2.imread(os.path.join(root,file))
                    if similarity_check(img1, img2, 0.3):
                        shutil.copy2(os.path.join(root,file), os.path.join(image_path,file))
                        img1 = img2
                else:
                    video_id = new_video_id
                    img1 = cv2.imread(os.path.join(root, file))
                    shutil.copy2(os.path.join(root,file), os.path.join(image_path,file))


                print(new_video_id)


                tbar.update(1)

                # cam = cv2.VideoCapture(os.path.join(root,file))
                # frameno=0
                # img1 = 0
                # while(1):
                #     ret,frame = cam.read()

                #     if ret:
                #         name = file.split(".")[0]+"_"+str(frameno)+".JPG"

                #         if frameno == 0 or similarity_check(img1, frame):
                #             img1 = frame
                #             cv2.imwrite(os.path.join(image_path,name), frame)
                #             # cv2.imshow(name, frame)
                #             # cv2.waitKey(0)
                #         frameno +=1
                #     else:
                #         break

                # cam.release()


def copy_annotation():

    annotation_folder = "F:\\pest_data\\unannotated\\2024\\Image_From_Video_0.35"
    image_path = "F:\\pest_data\\unannotated\\2024\\Image_From_Video_0.3"

    for root, folder, files in os.walk(image_path):
        for file in files:
            print()
            if os.path.exists(os.path.join(annotation_folder, ".".join(file.split(".")[0:-1])+".xml")):
                shutil.copy2(os.path.join(annotation_folder, ".".join(file.split(".")[0:-1])+".xml" ), os.path.join(image_path, ".".join(file.split(".")[0:-1])+".xml" ) )
                print("Copy annotation")

def move_annotated_images():
    original_folder = "F:\\pest_data\\unannotated\\Annotated_Videos"
    target_folder = "F:\\pest_data\\unannotated\\Manual_Checked_Video_Annotation"

    for root, folders, files in os.walk(original_folder):
        file_num = 0
        for file in files[::-1]:
            if file.split(".")[-1] == "xml" and file_num<5:
                if '.'.join([file.split(".")[0], "JPG"]) not in files:
                    continue
                new_folder = os.path.join(target_folder, root.split("\\")[-1])
                if not os.path.exists(new_folder):
                    os.mkdir(new_folder)
                shutil.copy2(os.path.join(root, file), os.path.join(new_folder, file))
                shutil.copy2(os.path.join(root, ".".join(file.split(".")[0:-1])+".JPG"), os.path.join(new_folder, ".".join(file.split(".")[0:-1])+".JPG"))
                file_num += 1

def remove_annotated_images():
    original_folder = "F:\\pest_data\\unannotated\\2024\\Image_From_Video_0.3"

    for root, folders, files in os.walk(original_folder):
        for file in files:
            if file.split(".")[-1] == "xml":
                os.remove(os.path.join(root, file))
                os.remove(os.path.join(root, ".".join(file.split(".")[0:-1])+".JPG"))


if __name__ == "__main__":

    # path="C:\\Users\\zhipeng\\Desktop\\14_Annotated\\14"

    # rename_file_type(path)

    # extract_img_from_video()

    # extract_img_from_img_folder()

    # copy_annotation()

    move_annotated_images()

    # remove_annotated_images()
