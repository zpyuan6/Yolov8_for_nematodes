import os
import xml.etree.ElementTree as ET

def rename_object():
    path="F:\\pest_data\\Multitask_or_multimodality\\remove_rotato"

    for r, folders, files in os.walk(path):
        for file in files:
            if file.split(".")[-1]=="xml":
                annotation_file = os.path.join(r, file)
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                objects = root.findall('object')

                for obj in objects:
                    if obj.find('name').text == "Insecta" or obj.find('name').text == "cabbage	aphid" or obj.find('name').text == "insecta":
                        obj.find('name').text = "Pollen beetle (Meligethes spp.)"
                    print(obj.find('name').text)

                tree.write(annotation_file)


if __name__=="__main__":
    rename_object()

