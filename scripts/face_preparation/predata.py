import os
import argparse
import re
import cv2
import shutil
from tqdm import *
# parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument("-k",
                    "--keepdims",
                    help="keep original image dimensions in downsampled images",
                    action="store_true")
parser.add_argument(
    "-w",
    "--wrap",
    help="Wrapping image data into a folder and scale must not be 'all' ",  #单独down一个比例，并且生成对应的gtmod文件夹
    action="store_true")
parser.add_argument('--hr_img_dir',
                    type=str,
                    default=r'E:\\PyProject\\data\\classical_SR_datasets\\gt\\Urban100\\GTmod',
                    help='path to high resolution image dir')
parser.add_argument('--lr_img_dir',
                    type=str,
                    default=r'E:\\PyProject\\data\\classical_SR_datasets\\gt\\Urban100',
                    help='path to desired output dir for downsampled images')
parser.add_argument('-s', "--scale", type=str, default=r'LRbicx4', help='scale of downsampled images || "all" scale')
parser.add_argument('--iteration', default=False, type=bool)

args = parser.parse_args()

hr_image_dir = args.hr_img_dir
lr_image_dir = args.lr_img_dir

print(f"GT_IMG_PATH:{args.hr_img_dir}")
print(f"LR_IMG_PATH:{args.lr_img_dir}")

supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras",
                         ".tif", ".tiff")


def downsample(filename, scale, hr_img, hr_img_dims, down_path):
    # cv2.GaussianBlur(hr_img, (0,0), 1, 1)
    # 其中模糊核这里用的0。两个1分别表示x、y方向的标准差。 可以具体查看该函数的官方文档。
    lr_image = cv2.resize(hr_img, (0, 0), fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_image = cv2.resize(lr_image, hr_img_dims, interpolation=cv2.INTER_CUBIC)
    if args.wrap:
        cv2.imwrite(
            # os.path.join(lr_image_dir + "/X2",
            #              filename.split('.')[0] + 'x2' + ext), lr_image_2x)
            os.path.join(wrap_dir, filename),
            hr_img)

    cv2.imwrite(
        # os.path.join(lr_image_dir + "/X2",
        #              filename.split('.')[0] + 'x2' + ext), lr_image_2x)
        os.path.join(down_path, filename),
        lr_image)


def process(scale, down_path, hr_image_dir):
    for filename in tqdm(os.listdir(hr_image_dir)):
        if not filename.endswith(supported_img_formats):
            continue

        name, ext = os.path.splitext(filename)

        # Read HR image
        hr_img = cv2.imread(os.path.join(hr_image_dir, filename))

        # Blur with Gaussian kernel of width sigma = 1
        hr_img = cv2.GaussianBlur(hr_img, (0, 0), 1, 1)
        if args.scale == 'all':
            for s, d in zip(scale, down_path):
                hr_img_dims = (hr_img.shape[1] // s * s, hr_img.shape[0] // s * s)
                hr_img = cv2.resize(hr_img, hr_img_dims)
                downsample(filename, s, hr_img, hr_img_dims, d)
        else:
            hr_img_dims = (hr_img.shape[1] // scale * scale, hr_img.shape[0] // scale * scale)
            hr_img = cv2.resize(hr_img, hr_img_dims)
            downsample(filename, scale, hr_img, hr_img_dims, down_path)
    print("process down")


def iteration(input_size=128, down_iter=3, datasetname='celeb'):
    dir_list = []
    print('输入图像大小：', input_size, '下采样次数：', down_iter, '数据集：', datasetname)
    for i in range(down_iter):
        print("down_iter", i+1)
        input_size = input_size / 2
        target_dir_path = os.path.join(lr_image_dir,datasetname + "_" + str(int(input_size)))
        os.makedirs(target_dir_path, exist_ok=True)
        dir_list.append(target_dir_path)# 64 32 16
        if i == 0:
            process(scale=2, down_path=target_dir_path, hr_image_dir=hr_image_dir)
        else:
            process(scale=2, down_path=target_dir_path, hr_image_dir=dir_list[i-1])



if __name__ == "__main__":

    # assert if args.wrap == True and args.scale == "all", "暂时不支持lr所有图像，并且封装原图像。"

    if args.scale == 'all':
        scale_list = []
        down_path_list = []
        scale_dir = ['LRbicx2', 'LRbicx3', 'LRbicx4', 'LRbicx8']
        for lr_dir in scale_dir:
            down_path = os.path.join(lr_image_dir, lr_dir)
            os.makedirs(down_path, exist_ok=True)
            scale_list.append(int(re.search("\d+", lr_dir).group()))
            down_path_list.append(down_path)
        process(scale_list, down_path_list, hr_image_dir)
    elif args.iteration == True:
        print('迭代下采样\n', '-'*50)
        iteration(input_size=128, down_iter=3, datasetname='celeb')
    else:
        scale = int(re.search("\d+", args.scale).group())
        assert scale > 0, "scale must be in 'LRbicX2','LRbicX4',..."
        down_path = os.path.join(lr_image_dir, args.scale)
        os.makedirs(down_path, exist_ok=True)
        wrap_dir = os.path.join(lr_image_dir, f"GTmod{scale}")
        if args.wrap:
            os.makedirs(wrap_dir, exist_ok=True)
        process(scale, down_path, hr_image_dir)

# python predata.py --scale LRbicx8 -w
# python predata.py --scale all
        
# python predata.py --hr_img_dir /home/ubuntu/workplace/pjh/data/train/celeb128 --lr_img_dir /home/ubuntu/workplace/pjh/data/train/celeb16_downx8 -s LRbicX8
        
# python predata.py --hr_img_dir /home/ubuntu/workplace/pjh/data/test/celeb128 --lr_img_dir /home/ubuntu/workplace/pjh/data/test/celeb16_downx8 -s LRbicX8
        
# python predata.py --hr_img_dir E:\PyProject\data\classical_SR_datasets\CelebA-HQ_ParsingMap\inference\GTmod128 --lr_img_dir E:\PyProject\data\classical_SR_datasets\CelebA-HQ_ParsingMap\inference --iteration True