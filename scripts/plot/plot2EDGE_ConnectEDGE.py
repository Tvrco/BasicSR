# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

'''
    face_full_boudnary 脸颊=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    face_left_eyebow ： 17,18,19,20,21
    face_right_eyebow： 22,23,24,25,26
    left_eye 左眼：36,37,38,39,40,41
    right_eye 右眼：42,43,44,45,46,47
    nose bridge 鼻子_up : 27,28，29,30
    nose_round 鼻孔：31,32,33,34,35
    嘴1：48,49,50,51,52,53,54,55,56,57,58,59,48
    嘴2：60,61,62,63,64
    嘴3：65,66,67
    牙齿：60,61,62,63,64,65,66,67


'''
# 源地址
# dir = "D:\\datasets_SR\\CelebAMask-HQ\\CelebA-LQ-img\\"
# 源地址
dir = "E:\\PyProject\\SR\\BasicSR_Server20231212\\BasicSR\\scripts\\plot\\face_1160"
# 目标地址
dest_dir_final = "E:\\PyProject\\SR\\BasicSR_Server20231212\\BasicSR\\scripts\\plot\\w_GB\\"

os.makedirs(dest_dir_final, exist_ok=True)

line_wight= 2

point_dst = [0,0]
point_src = [0,0]

face_full_boudnary_start = [0,0]
face_full_boudnary_end = [0,0]

face_left_eyebow_start = [0,0]
face_left_eyebow_end = [0,0]

face_right_eyebow_start = [0,0]
face_right_eyebow_end = [0,0]

nose_bridge_start = [0,0]
nose_bridge_end = [0,0]

nose_round_start = [0,0]
nose_round_end = [0,0]

left_eye_start = [0,0]
left_eye_end = [0,0]

right_eye_start = [0,0]
right_eye_end = [0,0]

upper_mouth_start = [0,0]
upper_mouth_end = [0,0]

upper_mouth_down_start = [0,0]
upper_mouth_down_end = [0,0]

lower_mouth_up_start = [0,0]
lower_mouth_up_end = [0,0]

lower_mouth_down_start = [0,0]
lower_mouth_down_end = [0,0]


class FaceLandmarksDataset(Dataset):
    """MY FACE LANDMARK DATASET"""

    def __init__(self, txt_file, root_dir, transform=None):
        # 从指定的 txt 文件中读取数据，存储到一个名为 landmarks_frame 的 pandas.DataFrame 对象中
        self.landmarks_frame = pd.read_csv(txt_file, header=None)
        # 指定数据文件所在的根目录
        self.root_dir = root_dir
        # 对数据进行预处理的可选变换参数
        self.transform = transform

    def __len__(self):
        # 定义数据集的长度
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # 定义获取单个样本的方法，传入参数 idx 表示数据集中的索引
        img_name = os.path.join(self.root_dir, self.landmarks_frame.loc[idx, 0])
        # print(self.landmarks_frame)
        # 从数据集的 landmarks_frame 中获取图片对应的文件名，并与 root_dir 拼接成完整路径
        image = (img_name)
        # 读取文件中的图片到内存中
        landmarks = self.landmarks_frame.iloc[idx,1:].values.astype('float')
        # print(f"landmarks:{landmarks}")
        # 从 landmarks_frame 中获取人脸关键点的坐标数据
        landmarks = landmarks.reshape(-1, 2)
        # 对关键点数据进行 reshape 操作，将一维数组转换成二维数组
        sample = {'image': image, 'landmarks': landmarks}
        print(sample['landmarks'].shape)
        # 构造样本对象，将图片及关键点坐标保存到 sample 字典中
        if self.transform:
            # 如果有指定可选变换，则进行数据增强操作
            sample = self.transform(sample)
        return sample

# face_dataset = FaceLandmarksDataset(txt_file ='D:\\datasets_SR\\CelebAMask-HQ\\landmarks.CSV',root_dir= dir)
# face_dataset = FaceLandmarksDataset(txt_file =txt_file,root_dir= dir)
face_dataset = FaceLandmarksDataset(txt_file ='E:\\PyProject\\SR\\BasicSR_Server20231212\\BasicSR\\scripts\\plot\\68_0905_landmarks.csv',root_dir= dir)


num_len = len(face_dataset)

for i in range(num_len):
    sample = face_dataset[i]
    print(i,sample['image'])
    (filepath,tempfilename) = os.path.split(sample['image'])
    (final_filename,extension) = os.path.splitext(tempfilename)


    count = 0
    img = cv2.imread(sample['image'])

    img_face_full_boudnary = np.zeros(img.shape)

    img_face_left_eyebow = np.zeros(img.shape)
    img_face_right_eyebow = np.zeros(img.shape)

    img_nose_bridge = np.zeros(img.shape)
    img_nose_round = np.zeros(img.shape)

    img_left_eye = np.zeros(img.shape)
    img_right_eye = np.zeros(img.shape)

    img_upper_mouth = np.zeros(img.shape)
    img_upper_mouth_down = np.zeros(img.shape)

    img_lower_mouth_up = np.zeros(img.shape)
    img_lower_mouth_down = np.zeros(img.shape)

    point_list = sample['landmarks']
    # 把点练成线的过程
    for point in point_list:
        count = count + 1

        # 这里的话拟采用四舍五入的方法，使得我们的生成的点更接近实际
        point[0] = round(point[0])
        point[1] = round(point[1])
        point = tuple(point)

        # cv2.circle(img,(int(point[0]),int(point[1])),1,(0,0,255),2)
        if count % 2 == 0:
            point_dst[0] = point[0]
            point_dst[1] = point[1]
        else:
            point_src[0] = point[0]
            point_src[1] = point[1]

        # extract face contour (from point 0 to 16)
        if count == 1:
            face_full_boudnary_start[0] = point[0]
            face_full_boudnary_start[1] = point[1]
            continue

        if count >= 1 and count <= 17:
            cv2.line(img_face_full_boudnary, (int(point_dst[0]), int(point_dst[1])),
                     (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 17:
            face_full_boudnary_end[0] = point[0]
            face_full_boudnary_end[1] = point[1]
            print("face contour end",face_full_boudnary_end)
            continue

        # extract left_eyebrow (from 17 to 21)
        if count == 18:
            face_left_eyebow_start[0] = point[0]
            face_left_eyebow_start[1] = point[1]
            print("this si the left eyebow start",face_left_eyebow_start)
            continue

        if count > 18 and count <= 22:
            cv2.line(img_face_left_eyebow, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 22:
            face_left_eyebow_end[0] = point[0]
            face_left_eyebow_end[1] = point[1]
            print("this is the left eyebow end",face_left_eyebow_end)
            cv2.line(img_face_left_eyebow, (int(face_left_eyebow_start[0]), int(face_left_eyebow_start[1])), (int(face_left_eyebow_end[0]), int(face_left_eyebow_end[1])),
                     (255, 255, 255), line_wight)
            continue

        # extract right_eyebrow (from 22 to 26)
        if count == 23:
            face_right_eyebow_start[0] = point[0]
            face_right_eyebow_start[1] = point[1]
            print("this is the right eyebow start",face_right_eyebow_start)
            continue

        if count >= 23 and count <= 27:
            cv2.line(img_face_right_eyebow, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 27:
            face_right_eyebow_end[0] = point[0]
            face_right_eyebow_end[1] = point[1]
            print("this is the right eyebow end",face_right_eyebow_end)
            cv2.line(img_face_right_eyebow, (int(face_right_eyebow_start[0]), int(face_right_eyebow_start[1])), (int(face_right_eyebow_end[0]), int(face_right_eyebow_end[1])),
                     (255, 255, 255), line_wight)
            continue

        # extract nose structure (from 27 to 30)
        if count == 28:
            nose_bridge_start[0] = point[0]
            nose_bridge_start[1] = point[1]
            print("this is nose start",nose_bridge_start)
            continue

        if count >= 28 and count <= 31:
            cv2.line(img_nose_bridge, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 31:
            nose_bridge_end[0] = point[0]
            nose_bridge_end[1] = point[1]
            print("this is the nose end",nose_bridge_end)
            continue

        # extract nose round (from 31 to 35)
        if count == 32:
            nose_round_start[0] = point[0]
            nose_round_start[1] = point[1]
            print("this is nose start",nose_round_start)
            continue

        if count >= 32 and count <= 36:
            cv2.line(img_nose_round, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 36:
            nose_round_end[0] = point[0]
            nose_round_end[1] = point[1]
            print("this is the nose end",nose_round_end)
            continue


        # extract left eye (from 36 to 41)
        if count == 37:
            left_eye_start[0] = point[0]
            left_eye_start[1] = point[1]
            print("this is nose start",left_eye_start)
            continue

        if count >= 37 and count <= 42:
            cv2.line(img_left_eye, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 42:
            left_eye_end[0] = point[0]
            left_eye_end[1] = point[1]
            print("this is the nose end",left_eye_end)
            cv2.line(img_left_eye, (int(left_eye_start[0]), int(left_eye_start[1])), (int(left_eye_end[0]), int(left_eye_end[1])),
                     (255, 255, 255), line_wight)
            continue

        # extract right eye (from 42 to 47)
        if count == 43:
            right_eye_start[0] = point[0]
            right_eye_start[1] = point[1]
            print("this is right eye start",right_eye_start)
            continue

        if count >= 43 and count <= 48:
            cv2.line(img_right_eye, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 48:
            right_eye_end[0] = point[0]
            right_eye_end[1] = point[1]
            print("this is right eye start",right_eye_start)
            cv2.line(img_right_eye, (int(right_eye_start[0]), int(right_eye_start[1])), (int(right_eye_end[0]), int(right_eye_end[1])),
                     (255, 255, 255), line_wight)
            continue

        # extract upper mouth up(from 48 to 54)
        if count == 49:
            upper_mouth_start[0] = point[0]
            upper_mouth_start[1] = point[1]
            print("this is the outer lips_start",upper_mouth_start)
            continue

        if count >= 49 and count <= 55:
            cv2.line(img_upper_mouth, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 55:
            upper_mouth_end[0] = point[0]
            upper_mouth_end[1] = point[1]
            print("this is the outer lips_end",upper_mouth_end)
            # cv2.line(img_upper_mouth, (int(upper_mouth_start[0]), int(upper_mouth_start[1])), (int(upper_mouth_end[0]), int(upper_mouth_end[1])),
            #          (255, 255, 255), line_wight)
            #continue

        # 将60点的判断同时写进同一个地方
        # extract upper mouth down(from 60 to 64)
        # extract lower mouth down(from 54 to 60)
        if count == 55:
            lower_mouth_down_start[0] = point[0]
            lower_mouth_down_start[1] = point[1]
            print("-----this is the lower_mouth_down_start", lower_mouth_down_start)
            # cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            continue

        if count > 55 and count <= 61:
            cv2.line(img_lower_mouth_down, (int(point_dst[0]), int(point_dst[1])),
                     (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 61:

            upper_mouth_down_start[0] = point[0]
            upper_mouth_down_start[1] = point[1]
            print("-this is the inner_lips_start",upper_mouth_down_start)

            lower_mouth_down_end[0] = point[0]
            lower_mouth_down_end[1] = point[1]
            print("-----this is the inner_lips_end", lower_mouth_down_end)
            # cv2.line(img_lower_mouth_down, (int(lower_mouth_down_start[0]), int(lower_mouth_down_start[1])),
            #          (int(lower_mouth_down_end[0]), int(lower_mouth_down_end[1])),
            #          (255, 255, 255), line_wight)
            continue

        if count >= 61 and count <= 65:
            cv2.line(img_upper_mouth_down, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 65:
            upper_mouth_down_end[0] = point[0]
            upper_mouth_down_end[1] = point[1]
            print("--this is the inner_lips_end",upper_mouth_down_end)
            # cv2.line(img_upper_mouth_down, (int(upper_mouth_down_start[0]), int(upper_mouth_down_start[1])), (int(upper_mouth_down_end[0]), int(upper_mouth_down_end[1])),
            #          (255, 255, 255), line_wight)
            continue

        # extract lower mouth up(from 65 to 67)
        if count == 66:
            lower_mouth_up_start[0] = point[0]
            lower_mouth_up_start[1] = point[1]
            print("---this is the inner_lips_start",lower_mouth_up_start)
            continue

        if count >= 66 and count <= 68:
            cv2.line(img_lower_mouth_up, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), line_wight)

        if count == 68:
            lower_mouth_up_end[0] = point[0]
            lower_mouth_up_end[1] = point[1]
            print("----this is the inner_lips_end",lower_mouth_up_end)
            # cv2.line(img_lower_mouth_up, (int(lower_mouth_up_start[0]), int(lower_mouth_up_start[1])), (int(lower_mouth_up_end[0]), int(lower_mouth_up_end[1])),
            #          (255, 255, 255), line_wight)
            continue





    print("Outputing edging map")
    ke = (5,5)
    img_face_full_boudnary = cv2.GaussianBlur(img_face_full_boudnary,ke,0)
    img_face_left_eyebow = cv2.GaussianBlur(img_face_left_eyebow,ke,0)
    img_face_right_eyebow = cv2.GaussianBlur(img_face_right_eyebow,ke,0)
    img_nose_bridge = cv2.GaussianBlur(img_nose_bridge,ke,0)
    img_nose_round = cv2.GaussianBlur(img_nose_round,ke,0)
    img_left_eye = cv2.GaussianBlur(img_left_eye,ke,0)
    img_right_eye = cv2.GaussianBlur(img_right_eye,ke,0)
    img_upper_mouth = cv2.GaussianBlur(img_upper_mouth,ke,0)
    img_upper_mouth_down = cv2.GaussianBlur(img_upper_mouth_down,ke,0)
    img_lower_mouth_up = cv2.GaussianBlur(img_lower_mouth_up,ke,0)
    img_lower_mouth_down = cv2.GaussianBlur(img_lower_mouth_down,ke,0)
    cv2.imwrite(dest_dir_final+final_filename+"_face_full_boundary.jpg",img_face_full_boudnary)
    cv2.imwrite(dest_dir_final+final_filename+"_face_left_eyebow.jpg",img_face_left_eyebow)
    cv2.imwrite(dest_dir_final+final_filename+"_face_right_eyebow.jpg",img_face_right_eyebow)
    cv2.imwrite(dest_dir_final+final_filename+"_face_nose_bridge.jpg",img_nose_bridge)
    cv2.imwrite(dest_dir_final+final_filename+"_face_nose_round.jpg",img_nose_round)
    cv2.imwrite(dest_dir_final+final_filename+"_face_left_eye.jpg",img_left_eye)
    cv2.imwrite(dest_dir_final+final_filename+"_face_right_eye.jpg",img_right_eye)
    cv2.imwrite(dest_dir_final+final_filename+"_face_upper_mouth.jpg",img_upper_mouth)
    cv2.imwrite(dest_dir_final+final_filename+"_face_upper_mouth_down.jpg",img_upper_mouth_down)
    cv2.imwrite(dest_dir_final+final_filename+"_face_lower_mouth_up.jpg",img_lower_mouth_up)
    cv2.imwrite(dest_dir_final+final_filename+"_face_lower_mouth_down.jpg",img_lower_mouth_down)



    #print(i,sample['image'].shape,sample['landmarks'].shape)


     # 定义文件名列表
    file_names = ["_face_full_boundary",
                  "_face_left_eyebow",
                  "_face_right_eyebow",
                  "_face_nose_bridge",
                  "_face_nose_round",
                  "_face_left_eye",
                  "_face_right_eye",
                  "_face_upper_mouth",
                  "_face_upper_mouth_down",
                  "_face_lower_mouth_up",
                  "_face_lower_mouth_down"]

    # 定义基准图像
    base_img = None

    # 循环遍历文件名列表，读取图像并叠加到基准图像上
    for name in file_names:
        #读取图像
        img = cv2.imread(dest_dir_final + final_filename + name + ".jpg")

        # 如果是第一个图像，则将其设为基准图像
        if base_img is None:
            base_img = img
        # 否则，将当前图像叠加到基准图像上
        else:
            cv2.addWeighted(base_img, 1, img, 1, 0, base_img)

    # 将叠加后的图像保存为新的图像
    cv2.imwrite(dest_dir_final + final_filename + "_all_faces.jpg", base_img)