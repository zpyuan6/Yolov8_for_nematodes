
import xml.etree.ElementTree as ET
import cv2
import os
 
# 读取 xml 文件信息，并返回字典形式
def parse_xml_to_dict(xml):
    if len(xml) == 0:  # 遍历到底层，直接返回 tag对应的信息
        return {xml.tag: xml.text}
 
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}
 
 
# xml 标注文件的可视化
def xmlShow(img,xml,save = True):
    image = cv2.imread(img)


    ob = []

    voc_annotation_file = open(xml, encoding='utf-8')
    tree = ET.parse(voc_annotation_file)
    root = tree.getroot()


    for i in root.iter('object'):

        name = str(i.find('name').text)        # 检测的目标类别
 
        bbox = i.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
 
        tmp = [name,xmin,ymin,xmax,ymax]    # 单个检测框
        ob.append(tmp)
 
    # 绘制检测框
    for name,x1,y1,x2,y2 in ob:
        cv2.rectangle(image,(x1,y1),(x2,y2),color=(255,0,0),thickness=2)    # 绘制矩形框
        cv2.putText(image,name,(x1,y1-10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,thickness=3,color=(0,0,255))
 
    # 保存图像
    if save:
        cv2.imwrite(img.split(".")[0]+'_result.png',image)
 
    # 展示图像
    # cv2.imshow('test',image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
 
 
if __name__ == "__main__":

    for root, folders, files in os.walk("C:\\Users\\zhipeng\\Desktop\\DataSample"):
        for file in files:
            if file.split(".")[-1] == "JPG":

                img_path = os.path.join(root, file) # 传入图片
            
                # labels_path = img_path.replace('images', 'labels')       # 自动获取对应的 xml 标注文件
                labels_path = img_path.replace('.JPG', '.xml')
            
                xmlShow(img=img_path, xml=labels_path,save=True)