import csv
from itertools import count
import random

import shutil
import numpy as np
import cv2
# import dlib
import pandas as pd
import os
from tqdm import tqdm

def read_img(full_filename):
    img = cv2.imread(full_filename)
    img = cv2.resize(img,(128,128))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img, img_gray

def count_faces(img):
    rects = detector(img, 0)
    count = len(rects)

    return rects, count


def save_imgs(img, dir, filename):
    img = cv2.imwrite(dir + filename, img)


# 先根据下面定义的Boundary Line作出11张处理过后的Face Boundary Line

def tocsv(root_dir,data_dir):
    filename = os.listdir(data_dir)
    filename.sort()
    num_count = 0
    for name in tqdm(filename):
        full_path = os.path.join(data_dir, name)
        img, img_gray = read_img(full_path)

        rects, num_faces = count_faces(img_gray)
        # print(num_faces)
        # 这一部分来保存提取到的x个人脸特征点
        if num_faces == 1:
            num_count +=1
            with open(os.path.join(root_dir,"landmarks—68-12.26.CSV"), "a+") as f:
                f.write(name)
                for i in range(num_faces):
                    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
                    for idx, point in enumerate(landmarks):
                        pos = (point[0, 0], point[0, 1])
                        f.write("," + str(pos[0]) + "," + str(pos[1]))
                        # print(idx, pos)
                f.write("\n")
            # cv2.imwrite(data_dir + name, img)
    print(f"共{len(filename)}张图像，成功识别{num_count}")


def moveFile(root_dir,fileDir, des_dir):
    os.makedirs(des_dir,exist_ok=True)
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)

    # 读取landmarks.csv中的文件名，保存到set中
    file_names = set()
    with open(os.path.join(root_dir,"landmarks.CSV"), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            file_names.add(row[0])
    print(filenumber,len(file_names))
    print(f"Copy {fileDir} to {des_dir}")
    for i, filename in tqdm(enumerate(pathDir), total=filenumber):
        if filename in file_names:
            # 如果文件名在set中，说明需要移动到trainDir文件夹
            src_path = os.path.join(fileDir, filename)
            dst_path = os.path.join(des_dir, filename)
            shutil.copyfile(src_path, dst_path)



def shuffle_moveFile(fileDir_HR,fileDir_LR, trainDir,dst_path_HR,dst_path_LR):
    dst_path_HR = os.path.join(trainDir,dst_path_HR)
    dst_path_LR = os.path.join(trainDir,dst_path_LR)
    os.makedirs(trainDir,exist_ok=True)
    os.makedirs(dst_path_HR,exist_ok=True)
    os.makedirs(dst_path_LR,exist_ok=True)
    pathDir = os.listdir(fileDir_HR)  # 取图片的原始路径
    filenumber = len(pathDir)
    # shuffle move
    rate1 = 0.8  # 自定义抽取csv文件的比例，比方说100张抽80个，那就是0.8
    rate1 = 0.8
    # picknumber1 = int(filenumber*rate1)  # 按照rate比例从文件夹中取一定数量的文件
    picknumber1 = int(17000)  # 按照rate比例从文件夹中取一定数量的文件
    sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
    print(f"shuffle Copy {fileDir_HR} to {dst_path_HR}")
    print(f"shuffle Copy {fileDir_LR} to {dst_path_LR}")
    for name in tqdm(sample1):
        shutil.copyfile(fileDir_HR + '/' + name, dst_path_HR + "/" + name)
        shutil.copyfile(fileDir_LR + '/' + name, dst_path_LR + "/" + name)

def shuffle_moveFile2(HRfileDir,LRfileDir,oriDir,dst_path):

    HRdir = os.path.join(oriDir,"GTmod128")
    LRdir = os.path.join(oriDir,"LRbicx2")
    dst_HRdir = os.path.join(dst_path,"GTmod128")
    dst_LRdir = os.path.join(dst_path,"LRbicx2")
    os.makedirs(dst_HRdir, exist_ok=True)
    os.makedirs(dst_LRdir, exist_ok=True)
    # 获取已经复制到trainDir文件夹中的图片文件名列表
    copied_filenames = os.listdir(HRdir) #17000
    copied_filenames = set(copied_filenames)

    pathDir = os.listdir(HRfileDir)  # 取图片的原始路径 #29021
    filenumber = len(pathDir)

    picknumber2 = 100  # 需要选取的图片数量
    count = 0
    while count < picknumber2:
        name = random.choice(pathDir)  # 随机选择一张图片
        if name not in copied_filenames:
            # 如果这张图片的文件名没有出现在之前复制的图片列表中，则将其复制到目标文件夹中
            shutil.copyfile(os.path.join(HRfileDir, name), os.path.join(dst_HRdir, name))
            shutil.copyfile(os.path.join(LRfileDir, name), os.path.join(dst_LRdir, name))
            count += 1
            print(f"Copied {name}")
        else:
            print(f"{name} already copied, select another image.")

    print(f"shuffle Copy {HRfileDir} to {dst_path}")


def shuffle_moveFile3(HRfileDir, LRfileDir, oriDir, dst_path):
    HRdir = os.path.join(oriDir, "GTmod128")
    LRdir = os.path.join(oriDir, "LRbicx2")
    dst_HRdir = os.path.join(dst_path, "GTmod128")
    dst_LRdir = os.path.join(dst_path, "LRbicx2")
    os.makedirs(dst_HRdir, exist_ok=True)
    os.makedirs(dst_LRdir, exist_ok=True)
    # 获取已经复制到trainDir文件夹中的图片文件名列表
    copied_filenames = os.listdir(HRdir)  # 17000
    copied_filenames = set(copied_filenames)

    pathDir = os.listdir(HRfileDir)  # 取图片的原始路径 #29021
    filenumber = len(pathDir)

    picknumber2 = 100  # 需要选取的图片数量
    count = 0
    for name in pathDir:
        if name not in copied_filenames:
            # 如果这张图片的文件名没有出现在之前复制的图片列表中，则将其复制到目标文件夹中
            shutil.copyfile(os.path.join(HRfileDir, name), os.path.join(dst_HRdir, name))
            shutil.copyfile(os.path.join(LRfileDir, name), os.path.join(dst_LRdir, name))
            count += 1
            print(f"Copied {name}")
        else:
            print(f"{name} already copied, select another image.")

    print(f"shuffle Copy {HRfileDir} to {dst_path}")
random.seed(42)
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks.dat')

root_dir = "E:\\PyProject\\SR\\BasicSR_Server20231212\\BasicSR\\scripts\\plot"

data_dir_GT = os.path.join(root_dir,"face_1160") #3w128*128
# data_dir_GT = os.path.join(root_dir,"GTmod128") #3w128*128
data_dir_LR = os.path.join(root_dir,"LRbicx2") #3w64*64
HR_dir = os.path.join(root_dir,"HR_face_dlib_GTmode128") #29021 128
LR_dir = os.path.join(root_dir,"HR_face_dlib_LRbicx2") #29021 64
trainDir = os.path.join(root_dir,"train") #随机筛选17000
testDir = os.path.join(root_dir,"test")
tocsv(root_dir,data_dir_GT)
#
# # 1.predata 下采样 1024-128
# # print("1.predata 下采样 1024-128")
# # os.system(f'python predata.py --scale LRbicx8  --hr_img_dir {os.path.join(root_dir,"CelebA-HQ-img")} --lr_img_dir {os.path.join(root_dir)}')
# # 1.5 手动将下采样的HR路径改成GTmod128
# print("LRbicx8 改名为 GTmod128")
# os.system(f'mv /content/1SFSRCeleb/data/celebA-H-Pansinglap/Rbicx8./content /MSFSR-Celeb/data/celebA-H0-Pansinglap/GTmod128')
# # 1.6 将128*128的GTmode下采样至64*64
# print("1.6 将128*128的GTmode下采样至64*64")
# os.system(f'python predata.py --scale LRbicx2--hr_img_dir {data_dir_GT} --ir_img_dir {os.path.join(root_dir)}')
#
# # 1.7 手动调整路径
# # 2.根据下采样HR图片dlib生成29021张图片的csv
# tocsv(root_dir,data_dir_GT) #3w-#29021
#
# # 3.根据能够识别的出来的图片复制到新文件夹#3w-#29021
# print("3.根据能够识别的出来的图片复制到新文件夹#3w-#29021")
# moveFile(root_dir,data_dir_GT,HR_dir)  #29021
# moveFile(root_dir, data_dir_LR,LR_dir) #29021
#
# # 4.从新文件夹随机获取17k到train，并随机分100张test，
# print("4.从新文件夹随机获取17k到train，并随机分12k张test，")
# shuffle_moveFile(HR_dir,LR_dir, trainDir,"GTmod128","LRbicx2") #17000 128 #1700064
# shuffle_moveFile3(HR_dir,LR_dir,trainDir,testDir)#100账test

# 5.用plot2生成边界128*128,plot2EDGE_ConnectEDGE.py的路径提前写好
# os.system('python plot2EDGE_ConnectEDGE.py')
# # 6.再将生成的边界图片下采样
# print("boundary 128*128 --> 64*64 ")
# os.system(f'python predata.py --scale LRbicx2  --hr_img_dir {os.path.join(trainDir,"FBGTmod128")} --lr_img_dir {os.path.join(trainDir,"FBLRbicx2")}')
# 6.5用plot2生成边界test 64*64,plot2EDGE_ConnectEDGE.py的路径提前写好
# os.system('python plot2EDGE_ConnectEDGE.py')
# # 7.FBLRbicx2目录还得手动移动下

# print("8.predata 下采样test-LRbicx8")
# os.system(f'python predata.py --scale LRbicx8  --hr_img_dir {os.path.join(root_dir,"test/GTmod128")} --lr_img_dir {os.path.join(root_dir,"test")}')
### 处理第一、第二阶段，实验只在第三阶段传递损失，因此只需128*128下采样16*16
# os.system(f'python predata.py --scale LRbicx4  --hr_img_dir {os.path.join(trainDir,"GTmod128")} --lr_img_dir {os.path.join(trainDir)}')