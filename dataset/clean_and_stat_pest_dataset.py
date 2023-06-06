import os
import shutil
import hashlib
import cv2
from matplotlib import pyplot as plt
import tqdm

def stat_images_and_annotation(root_folder):
    file_stat = {}
    xml_list = []
    for root, folders, files in os.walk(root_folder):
        for file in files:
            file_name,file_type = os.path.splitext(file)

            if file_type in file_stat:
                file_stat[file_type] += 1
            else:
                file_stat[file_type] = 1

            # if file_type == ".xml":
            #     if file_name in xml_list:
            #         print(os.path.join(root,file))
            #     else:
            #         xml_list.append(file_name)
    
    print(file_stat)

IMAGE_TYPE = ['.jpg', '.JPG', '.JPEG', '.PNG', '.heic']

def transfer_file_to_build_dataset(root_folder, target_folder):
    def copy_file(src_path, obj_path, file_name, file_type):
        if os.path.exists(os.path.join(target_folder,file_name+file_type)):
            tag = root.split("\\")[-1]
            shutil.copy(os.path.join(root,file), os.path.join(target_folder,file_name+f"_{tag}"+file_type))
        else:
            shutil.copy(os.path.join(root,file), os.path.join(target_folder,file_name+file_type))


    for root, folders, files in os.walk(root_folder):
        for file in files:
            file_name,file_type = os.path.splitext(file)
            # get annotation file
            if file_type == ".xml":

                copy_file(root, target_folder, file_name,file_type)

                not_finded_file = True
                for optional_img_type in IMAGE_TYPE:
                    if file_name+optional_img_type in files:
                        copy_file(root, target_folder, file_name,optional_img_type)
                        not_finded_file = False
                        break
                
                if not_finded_file:
                    root_list = root.split("\\")
                    if "VOC2007" in root_list:
                        print("Please copy file for annotation: ", os.path.join(root, file))
                    else:
                        print("can not find img file for annotation: ", os.path.join(root, file))


#均值哈希算法
def aHash(img):
    #缩放为8*8
    img=cv2.resize(img,(30,30),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str

def check_duplicate_image(path):
    image_hashcode = []
    duplicate_image = []
    for root, folders, files in os.walk(path):
        with tqdm.tqdm(total=len(files)/2) as tbar:
            for file in files:
                if file.split(".")[-1]=="jpg" or file.split(".")[-1]=="JPG":
                    img = plt.imread(os.path.join(root, file))
                    hash = aHash(img)
                    if hash in image_hashcode:
                        duplicate_image.append(file)
                    else:
                        image_hashcode.append(hash)
                
                tbar.update(1)
                tbar.set_description(f"duplicate num: {len(duplicate_image)}")
    
    print(duplicate_image)
    
def delete_duplicate_image(path):
    duplicate = ['IMG_20220519_103327.jpg', 'IMG_20220519_103408.jpg', 'IMG_20220519_104141.jpg', 'IMG_20220519_104244.jpg', 'IMG_20220519_104502.jpg', 'IMG_20220519_104833.jpg', 'IMG_20220519_105049.jpg', 'IMG_20220519_105057.jpg', 'IMG_20220519_105104.jpg', 'IMG_20220519_105254.jpg', 'IMG_20220519_105330.jpg', 'IMG_20220519_105342.jpg', 'IMG_20220519_105420.jpg', 'IMG_20220519_105912.jpg', 'IMG_20220519_105916.jpg', 'IMG_20220519_105922.jpg', 'IMG_20220519_110109.jpg', 'IMG_20220519_110114.jpg', 'IMG_20220519_110159.jpg', 'IMG_20220519_110216.jpg', 'IMG_20220519_110220.jpg', 'IMG_20220519_110332.jpg', 'IMG_20220519_110533.jpg', 'IMG_20220519_110617.jpg', 'IMG_20220519_110647.jpg', 'IMG_20220519_110658.jpg', 'IMG_20220519_110834.jpg', 'IMG_20220519_110844.jpg', 'IMG_20220519_110926.jpg', 'IMG_20220519_110929.jpg', 'IMG_20220519_110935.jpg', 'IMG_20220519_111107.jpg', 'IMG_20220519_111230.jpg', 'IMG_20220519_111238.jpg', 'IMG_20220519_111304.jpg', 'IMG_20220519_111324.jpg', 'IMG_20220519_111330.jpg', 'IMG_20220519_111607.jpg', 'IMG_20220519_111611.jpg', 'IMG_20220519_111632.jpg', 'IMG_20220519_111655.jpg', 'IMG_20220519_111720.jpg', 'IMG_20220519_111824.jpg', 'IMG_20220519_111844.jpg', 'IMG_20220519_111951.jpg', 'IMG_20220519_112027.jpg', 'IMG_20220519_112037.jpg', 'IMG_20220519_112043.jpg', 'IMG_20220519_112058.jpg', 'IMG_20220519_112306.jpg', 'IMG_20220519_112329.jpg', 'IMG_20220519_112339.jpg', 'IMG_20220519_112343.jpg', 'IMG_20220519_112402.jpg', 'IMG_20220519_112837.jpg', 'IMG_20220519_112840.jpg', 'IMG_20220519_112923.jpg', 'IMG_20220519_113003.jpg', 'IMG_20220519_113005.jpg', 'IMG_20220519_113109.jpg', 'IMG_20220519_113119.jpg', 'IMG_20220519_113229.jpg', 'IMG_20220519_113238.jpg', 'IMG_20220519_113244.jpg', 'IMG_20220519_113410.jpg', 'IMG_20220519_113415.jpg', 'IMG_20220519_113723.jpg', 'IMG_20220519_113727.jpg', 'IMG_20220519_113733.jpg', 'IMG_20220519_114014.jpg', 'IMG_20220519_114020.jpg', 'IMG_20220519_114047.jpg', 'IMG_20220519_114102.jpg', 'IMG_20220519_114233.jpg', 'IMG_20220519_114237.jpg', 'IMG_20220519_114334.jpg', 'IMG_20220519_114527.jpg', 'IMG_20220519_114728.jpg', 'IMG_20220519_114736.jpg', 'IMG_20220519_114939.jpg', 'IMG_20220519_115255.jpg', 'IMG_20220519_115656.jpg', 'IMG_20220519_115751.jpg', 'IMG_20220519_115804.jpg', 'IMG_20220519_115808.jpg', 'IMG_20220519_115821.jpg', 'IMG_20220519_115840.jpg', 'IMG_20220519_115845.jpg', 'IMG_20220519_115901.jpg', 'IMG_20220519_115935.jpg', 'IMG_20220519_115948.jpg', 'IMG_20220519_120045.jpg', 'IMG_20220519_120117.jpg', 'IMG_20220519_120244.jpg', 'IMG_20220519_142414.jpg', 'IMG_20220519_142457.jpg', 'IMG_20220519_142831.jpg', 'IMG_20220519_142905.jpg', 'IMG_20220519_143103.jpg', 
'IMG_20220519_143120.jpg', 'IMG_20220519_143434.jpg', 'IMG_20220519_143515.jpg', 'IMG_20220519_143525.jpg', 'IMG_20220519_143531.jpg', 'IMG_20220519_143557.jpg', 'IMG_20220519_143608.jpg', 'IMG_20220519_144005.jpg', 'IMG_20220519_144244.jpg', 'IMG_20220519_144341.jpg', 'IMG_20220519_144435.jpg', 'IMG_20220519_144454.jpg', 'IMG_20220519_144458.jpg', 'IMG_20220519_144710.jpg', 'IMG_20220519_144717.jpg', 'IMG_20220519_145036.jpg', 'IMG_20220519_145316.jpg', 'IMG_20220519_145424.jpg', 'IMG_20220519_145747.jpg', 'IMG_20220519_145826.jpg', 'IMG_20220519_145901.jpg', 'IMG_20220519_145904.jpg', 'IMG_20220519_145906.jpg', 'IMG_20220519_145919.jpg', 'IMG_20220519_150403.jpg', 'IMG_20220519_150558.jpg', 'IMG_20220519_150608.jpg', 'IMG_20220519_150712.jpg', 'IMG_20220519_150743.jpg', 'IMG_20220519_150747.jpg', 'IMG_2651.JPG', 'IMG_2652.JPG', 'IMG_2653.JPG', 'IMG_2654.JPG', 'IMG_2655.JPG', 'IMG_2656.JPG', 'IMG_2657.JPG', 'IMG_2659.JPG', 'IMG_2660.JPG', 'IMG_2661.JPG', 'IMG_2662.JPG', 'IMG_2663.JPG', 'IMG_2664.JPG', 'IMG_2665.JPG', 'IMG_2666.JPG', 'IMG_2667.JPG', 'IMG_2668.JPG', 'IMG_2669.JPG', 'IMG_2670.JPG', 'IMG_2671.JPG', 'IMG_2672.JPG', 
'IMG_2673.JPG', 'IMG_2683.JPG', 'IMG_2684.JPG', 'IMG_2685.JPG', 'IMG_2686.JPG', 'IMG_2689.JPG', 'IMG_2690.JPG', 'IMG_2691.JPG', 'IMG_2692.JPG', 'IMG_2693.JPG', 'IMG_2694.JPG', 'IMG_2695.JPG', 'IMG_2696.JPG', 'IMG_2697.JPG', 'IMG_2698.JPG', 'IMG_2699.JPG', 'IMG_2700.JPG', 'IMG_2701.JPG', 'IMG_2702.JPG', 'IMG_2703.JPG', 'IMG_2704.JPG', 'IMG_2705.JPG', 'IMG_2706.JPG', 'IMG_2707.JPG', 'IMG_2708.JPG', 'IMG_2709.JPG', 'IMG_2710.JPG', 'IMG_3046.JPG', 'IMG_3049.JPG', 'IMG_3050.JPG', 'IMG_3051.JPG', 'IMG_3052.JPG', 'IMG_3054.JPG', 'IMG_3055.JPG', 'IMG_3056.JPG', 'IMG_3057.JPG', 'IMG_3058.JPG', 'IMG_3059.JPG', 'IMG_3060.JPG', 'IMG_3061.JPG', 'IMG_3062.JPG', 'IMG_3063.JPG', 'IMG_3064.JPG', 'IMG_3065.JPG', 'IMG_3066.JPG', 'IMG_3067.JPG', 'IMG_3070.JPG', 'IMG_3071.JPG', 'IMG_3072.JPG', 'IMG_3073.JPG', 'IMG_3074.JPG', 'IMG_3075.JPG', 'IMG_3077.JPG', 'IMG_3078.JPG', 'IMG_3079.JPG', 'IMG_3080.JPG', 'IMG_3082.JPG', 'IMG_3083.JPG', 'IMG_3084.JPG', 'IMG_3085.JPG', 'IMG_3086.JPG', 'IMG_3087.JPG', 'IMG_3088.JPG', 'IMG_3089.JPG', 'IMG_3090.JPG', 'IMG_3091.JPG', 'IMG_3092.JPG', 'IMG_3093.JPG', 'IMG_3094.JPG', 'IMG_3095.JPG', 'IMG_3096.JPG', 'IMG_3097.JPG', 'IMG_3098.JPG', 'IMG_3099.JPG', 'IMG_3100.JPG', 'IMG_3101.JPG', 'IMG_3103.JPG', 'IMG_3104.JPG', 'IMG_3105.JPG', 'IMG_3106.JPG', 'IMG_3107.JPG', 'IMG_3108.JPG', 'IMG_3109.JPG', 'IMG_3111.JPG', 'IMG_3113.JPG', 'IMG_3114.JPG', 'IMG_3247.jpg', 'IMG_3250.jpg', 'IMG_3251.jpg', 'IMG_3253.jpg', 'IMG_3254.jpg', 'IMG_3255.jpg', 'IMG_3256.jpg', 'IMG_3257.jpg', 'IMG_3258.jpg', 'IMG_3259.jpg', 'IMG_3260.jpg', 'IMG_3261.jpg', 'IMG_3262.jpg', 'IMG_3263.jpg', 'IMG_3265.jpg', 'IMG_3266.jpg', 'IMG_3267.jpg', 'IMG_3268.jpg', 'IMG_3269.jpg', 'IMG_3270.jpg', 'IMG_3271.jpg', 'IMG_3272.jpg', 'IMG_3273.jpg', 'IMG_3274.jpg', 'IMG_3277.jpg', 'IMG_3278.jpg', 'IMG_3280.jpg', 'IMG_3282.jpg', 'IMG_3283.jpg', 'IMG_3284.jpg', 'IMG_3285.jpg', 'IMG_3286.jpg', 'IMG_3287.jpg', 'IMG_3288.jpg', 'IMG_3289.jpg', 'IMG_3290.jpg', 'IMG_3291.jpg', 'IMG_3292.jpg', 'IMG_3293.jpg', 'IMG_3294.jpg', 'IMG_3295.jpg', 'IMG_3296.jpg', 'IMG_3297.jpg', 'IMG_3298.jpg', 'IMG_3299.jpg', 'IMG_4146.jpg', 'IMG_4147.jpg', 'IMG_4148.jpg', 'IMG_4149.jpg', 'IMG_4150.jpg', 'IMG_4151.jpg', 'IMG_4152.jpg', 'IMG_4153.jpg', 'IMG_4156.jpg', 'IMG_4158.jpg', 'IMG_4159.jpg', 'IMG_4160.jpg', 'IMG_4161.jpg', 'IMG_4166.jpg', 'IMG_4168.jpg', 'IMG_4169.jpg', 'IMG_4170.jpg', 'IMG_4171.jpg', 'IMG_4172.jpg', 'IMG_4174.jpg', 'IMG_4180.jpg', 'IMG_4299.jpg', 'IMG_4300.jpg', 'IMG_4301.jpg', 'IMG_4302.jpg', 'IMG_4303.jpg', 'IMG_4304.jpg', 'IMG_4305.jpg', 'IMG_4306.jpg', 'IMG_4307.jpg', 'IMG_4308.jpg', 'IMG_4309.jpg', 'IMG_4312.jpg', 'IMG_4313.jpg', 'IMG_4314.jpg', 'IMG_4315.jpg', 'IMG_4316.jpg', 'IMG_4317.jpg', 'IMG_4318.jpg', 'IMG_4319.jpg', 'IMG_4320.jpg', 'IMG_4321.jpg', 'IMG_4322.jpg', 'IMG_4323.jpg', 'IMG_4324.jpg', 'IMG_4327.jpg', 'IMG_4328.jpg', 'IMG_4329.jpg', 'IMG_4330.jpg', 'IMG_4331.jpg', 'IMG_4332.jpg', 'IMG_4333.jpg', 'IMG_4334.jpg', 'IMG_4335.jpg', 'IMG_4336.jpg', 'IMG_4337.jpg', 'IMG_4338.jpg', 'IMG_4339.jpg', 'IMG_4340.jpg', 'IMG_4341.jpg', 
'IMG_4342.jpg', 'IMG_4343.jpg', 'IMG_4344.jpg', 'IMG_4345.jpg', 'IMG_4346.jpg', 'IMG_4347.jpg', 'IMG_4348.jpg', 'IMG_4349.jpg', 'IMG_4350.jpg', 'IMG_4351.jpg', '_DSC4233.jpg', '_DSC4234.jpg', '_DSC4235.jpg', '_DSC4236.jpg', '_DSC4237.jpg', '_DSC4238.jpg', '_DSC4239.jpg', '_DSC4240.jpg', '_DSC4241.jpg', '_DSC4242.jpg', '_DSC4243.jpg', '_DSC4244.jpg', '_DSC4245.jpg', '_DSC4247.jpg', '_DSC4248.jpg', '_DSC4249.jpg', '_DSC4250.jpg', '_DSC4251.jpg', '_DSC4252.jpg', '_DSC4253.jpg', '_DSC4254.jpg', '_DSC4255.jpg', '_DSC4256.jpg', '_DSC4259.jpg', '_DSC4260.jpg', '_DSC4261.jpg', '_DSC4262.jpg', '_DSC4264.jpg', '_DSC4265.jpg', '_DSC4267.jpg', '_DSC4269.jpg', '_DSC4270.jpg', '_DSC4271.jpg', '_DSC4272.jpg', '_DSC4273.jpg', '_DSC4274.jpg', '_DSC4278.jpg', '_DSC4279.jpg', '_DSC4280.jpg', '_DSC4281.jpg']

    for root, folders, files in os.walk(path):
        for file in files:
            if file in duplicate:
                os.remove(os.path.join(root, file))
                annotation_file = file.split(".")[0]+".xml"
                os.remove(os.path.join(root, annotation_file))


                



if __name__ == "__main__":
    # root_folder = "F:\\Pest\\pest_data\\UK_Pest_Original"
    # stat_images_and_annotation(root_folder)

    # target_folder = "F:\Pest\pest_data\Pest_Dataset_2023"
    # transfer_file_to_build_dataset(root_folder, target_folder)
    # stat_images_and_annotation(target_folder)

    path = "F:\Pest\pest_data\Dataset-2022\Annotated_Data"

    # check_duplicate_image(path)

    delete_duplicate_image(path)
