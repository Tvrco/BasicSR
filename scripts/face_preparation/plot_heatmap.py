import cv2
import numpy as np
from matplotlib import pyplot as plt

# 假设 keypoints 是一个包含人脸关键点位置的列表，每个关键点是 (x, y) 格式
keypoints = [(5, 5)]

# 创建一个空白图像
height, width = 128, 128  # 根据需要调整图像大小
image = np.zeros((height, width), np.float32)
image = cv2.imread("1160_HR.jpg")
# 在图像上标记关键点位置
for point in keypoints:
    cv2.circle(image, point, 5, (255), -1)  # 使用小圆圈标记关键点位置

# 使用高斯模糊生成热力图效果
img = cv2.GaussianBlur(image, (0, 0), 10)


# COLORMAP_AUTUMN = 0,
# COLORMAP_BONE = 1,
# COLORMAP_JET = 2,
# COLORMAP_WINTER = 3,
# COLORMAP_RAINBOW = 4,
# COLORMAP_OCEAN = 5,
# COLORMAP_SUMMER = 6,
# COLORMAP_SPRING = 7,
# COLORMAP_COOL = 8,
# COLORMAP_HSV = 9,
# COLORMAP_PINK = 10,
# COLORMAP_HOT = 11

# img = cv2.imread("1160_HR.jpg")
for i in range(0, 13):
    im_color = cv2.applyColorMap(img, i)
    # 显示热力图
    plt.imshow(im_color, cmap='hot')
    plt.show()
    cv2.imwrite("test\\{}.jpg".format(i), im_color)