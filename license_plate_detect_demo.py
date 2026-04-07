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
filtered = cv2.bilateralFilter(gray, 13, 15, 15)

# show_image("gray", filtered)

# 边缘检测 canny边缘检测 第一个参数：输入图像 第二个参数：低阈值 第三个参数：高阈值
edges = cv2.Canny(filtered, 30, 200)

# show_image("edges", edges)

# 寻找轮廓 CHAIN_APPROX_SIMPLE保留轮廓的端点信息
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 找到前10个contourArea轮廓面积和排序
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

# 画轮廓
cv2.drawContours(img, contours, -1, [0, 255, 0])

show_image("img", img)