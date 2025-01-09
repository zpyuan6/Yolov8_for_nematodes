import os
import xml.etree.ElementTree as ET
from PIL import Image
import random
from tqdm import tqdm

def find_image_file(folder, file_name):

    for filename in os.listdir(folder):
        if filename.startswith(file_name):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.jpeg'):
                return os.path.join(folder, filename)

    return None

def cutting_images_from_xml_annotation(
    annoated_image_folder,
    target_classification_folder
    ):

    if not os.path.exists(target_classification_folder):
        os.makedirs(target_classification_folder)
    
    for filename in os.listdir(annoated_image_folder):
        if not filename.endswith('.xml'):
            continue

        xml_path = os.path.join(annoated_image_folder, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_path = find_image_file(annoated_image_folder, filename.replace('.xml', ''))

        i=0

        for obj in root.iter('object'):

            class_name = obj.find('name').text.upper()
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            try:
                with Image.open(image_path) as im:
                    cropped_img  = im.crop((xmin, ymin, xmax, ymax))
                    
                    class_dir = os.path.join(target_classification_folder, class_name)
                    if not os.path.exists(class_dir):
                        os.makedirs(class_dir)

                    save_path = os.path.join(class_dir, f"{filename.replace('.xml', '')}_id_{i}_position_{xmin}_{ymin}_{xmax}_{ymax}.jpg")
                    cropped_img.save(save_path)
                    print(f"Saved: {save_path}")

            except Exception as e:
                print(f"Error: {e}")
                continue

            i += 1

def cutting_background_images_based_on_xml_annotation(
    annoated_image_folder,
    target_classification_folder
    ):

    target_classification_folder = os.path.join(target_classification_folder, "background")

    if not os.path.exists(target_classification_folder):
        os.makedirs(target_classification_folder)

    number_images = 0

    with tqdm(total=5000) as pbar:
        for filename in os.listdir(annoated_image_folder):
            if not filename.endswith('.xml'):
                continue

            xml_path = os.path.join(annoated_image_folder, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_path = find_image_file(annoated_image_folder, filename.replace('.xml', ''))

            i=0

            im = Image.open(image_path)

            y = im.size[1]
            x = im.size[0]

            for obj in root.iter('object'):

                class_name = obj.find('name').text.upper()
                bbox = obj.find('bndbox')

                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                x_shift = random.randint(-xmin, x - xmax)
                y_shift = random.randint(-ymin, y - ymax)

                xmin = xmin + x_shift
                ymin = ymin + y_shift
                xmax = xmax + x_shift
                ymax = ymax + y_shift

                try:
                    cropped_img  = im.crop((xmin, ymin, xmax, ymax))

                    save_path = os.path.join(target_classification_folder, f"{filename.replace('.xml', '')}_id_{i}_position_{xmin}_{ymin}_{xmax}_{ymax}.jpg")
                    cropped_img.save(save_path)
                    print(f"Saved: {save_path}")

                    number_images += 1

                    pbar.update(1)

                    break

                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            if number_images > 5000:
                break

    

if __name__ == "__main__":
    annoated_image_folder = "F:\\pest_data\\Multitask_or_multimodality\\annotated_data"
    target_classification_folder = "X:\\pervasive_group\\PestProject\\classification_dataset"

    # cutting_images_from_xml_annotation(
    #     annoated_image_folder,
    #     target_classification_folder
    # )

    cutting_background_images_based_on_xml_annotation(
        annoated_image_folder,
        target_classification_folder
    )