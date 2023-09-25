import cv2
import os
# Ignore warnings
from tqdm import trange,tqdm
def all_face_fusion(gt_ori_path,face_split_path,target_path):
    '''
    ori_path必须为正常GT的路径
    '''
    ori_path_list = os.listdir(gt_ori_path)
    # 将叠加后的图像保存为新的图像
    face_boudnary_dir = os.path.join(target_path, 'face_boundary')
    os.makedirs(face_boudnary_dir,exist_ok=True)

    for ori_img in tqdm(ori_path_list):
        final_filename = ori_img.split('.')[0]
        # print(f'final_filename:{final_filename}')
            # 定义文件名列表
        file_names = ["_full_boundary",
                    "_left_eyebow",
                    "_right_eyebow",
                    "_nose_bridge",
                    "_nose_round",
                    "_left_eye",
                    "_right_eye",
                    "_upper_mouth",
                    "_upper_mouth_down",
                    "_lower_mouth_up",
                    "_lower_mouth_down"]
        # # 循环保存图片
        # for name in file_names:
        #     print(["img" + name])
        #     cv2.imwrite(dest_dir_final + "/" + final_filename + name + ".jpg", globals()["img" + name])
        # 定义基准图像
        base_img = None
        # base_img = cv2.imread(dir + "/" + final_filename  + ".jpg")

        # 循环遍历文件名列表，读取图像并叠加到基准图像上
        for name in file_names:
            # print("-"*50)
            # print(dest_dir_final + "/" + final_filename  + name + ".jpg")
            #读取图像
            img = cv2.imread(face_split_path + "/" + final_filename  + name + ".jpg")

            # 如果是第一个图像，则将其设为基准图像
            if base_img is None:
                base_img = img
            # 否则，将当前图像叠加到基准图像上
            else:
                cv2.addWeighted(base_img, 1, img, 1, 0, base_img)
        cv2.imwrite(face_boudnary_dir + "/" + final_filename + ".jpg", base_img)
gt_ori_path = '/home/ubuntu/workplace/pjh/data/train/celeb128'
face_split_path = '/home/ubuntu/workplace/pjh/data/celeb_fb128_wight2'
target_path = '/home/ubuntu/workplace/pjh/data'
# all_face_fusion(gt_ori_path,face_split_path,target_path)

import re

# 指定目标目录
target_directory = "/home/ubuntu/workplace/pjh/data/face_boundary"

# 遍历目录下的文件
for filename in os.listdir(target_directory):
    # 使用正则表达式匹配带有_boundary的部分并替换为空字符串
    new_filename = re.sub(r'_boundary', '', filename)
    
    # 构建新的文件路径
    old_path = os.path.join(target_directory, filename)
    new_path = os.path.join(target_directory, new_filename)
    
    # 重命名文件
    os.rename(old_path, new_path)

print("重命名完成")


