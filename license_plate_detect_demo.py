import cv2
import numpy as np

def show_image(desc, image):
    cv2.imshow(desc, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
加载图像和预处理
'''
# 加载图像
img = cv2.imread("test.jpg")
if img is None:
    print("无法加载图像，请检查文件路径")
    exit()

# 调整图片大小
img = cv2.resize(img, (1024, 800))

# 转成灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
双边滤波，保留边缘细节的同时还可以去除背景噪声
第一个参数：输入图像
第二个参数：领域直径
第三个参数：颜色空间的标准差
第四个参数：坐标空间的标准差
'''
# 正确的双边滤波调用
filtered = cv2.bilateralFilter(gray, 9, 75, 75)

show_image("gray", filtered)