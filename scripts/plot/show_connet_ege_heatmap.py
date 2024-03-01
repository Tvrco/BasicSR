import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Load the CSV file containing the facial landmarks
landmarks_path = '68_0905_landmarks.CSV'
landmarks_df = pd.read_csv(landmarks_path)

# Load the image
image_path = './face_1160/1160.jpg'
output_path = './1160_with_landmarks.jpg'

import numpy as np

# Manually extracted landmarks from the user's message
landmarks = np.array([
    (30, 58), (31, 68), (32, 77), (34, 86), (37, 94), (42, 102), (48, 109), (55, 115),
    (63, 117), (72, 116), (80, 110), (87, 103), (93, 95), (96, 86), (98, 76), (100, 67),
    (102, 56), (32, 54), (37, 51), (43, 51), (49, 52), (55, 55), (69, 55), (75, 52),
    (82, 50), (89, 51), (95, 54), (62, 62), (62, 69), (62, 76), (61, 83), (56, 86),
    (59, 87), (62, 88), (66, 87), (69, 85), (40, 61), (44, 59), (50, 59), (54, 63),
    (49, 64), (44, 64), (72, 63), (76, 59), (82, 59), (86, 62), (82, 64), (77, 64),
    (49, 94), (54, 94), (59, 93), (63, 94), (66, 93), (72, 94), (78, 94), (72, 101),
    (67, 103), (63, 103), (59, 103), (54, 101), (52, 95), (59, 96), (63, 96), (66, 96),
    (75, 95), (67, 99), (63, 99), (59, 99)
])

# Re-load the image
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct color plotting
height, width = 128, 128  # 根据需要调整图像大小
image = np.zeros((height, width), np.float32)
# Convert hex color to RGB
color = tuple(int('28A8F4'[i:i+2], 16) for i in (0, 2, 4))
# Draw the landmarks as dots on the image
for x, y in landmarks:
    cv2.circle(image, (x, y), 2, color, -1)
heatmap = cv2.GaussianBlur(image, (0, 0), 2)
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# 应用颜色映射
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 显示热力图
cv2.imshow('Heatmap', colored_heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("1160_heatmap.png",colored_heatmap)