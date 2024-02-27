import numpy as np
import cv2

# 假设 keypoints 是一个包含人脸关键点位置的列表，每个关键点是 (x, y) 格式
keypoints = [(30, 50)]

# 创建一个空白图像
height, width = 128, 128  # 根据需要调整图像大小
image = np.zeros((height, width), np.float32)  # 用float32来创建图像以适用于applyColorMap

# 在图像上标记关键点位置
for point in keypoints:
    cv2.circle(image, point, 5, (1), -1)  # 使用小圆圈标记关键点位置

# 使用高斯模糊生成热力图效果
heatmap = cv2.GaussianBlur(image, (0, 0), 3)

# 归一化热力图
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# 应用颜色映射
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 显示热力图
cv2.imshow('Heatmap', colored_heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("colored_heatmap.png",colored_heatmap)
