import cv2
import numpy as np
from paddleocr import TextRecognition

screenCnt = None

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

'''
判断主色调
'''
def reg_area_color(image):
    #BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #开运算（先腐蚀再膨胀） 去除小的干扰块，保留大的色块
    kernel = np.ones((35, 35), np.uint8)
    open = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)

    #直方图（对图像的H通道进行色调统计，看看哪个颜色出现的多） [opem]输入图像 H 0- 179
    hist = cv2.calcHist([open], [0], None, [180], [0, 180])

    # 找到最大的像素数量
    hist_max = np.where(hist == np.max(hist))

    # 判断颜色范围
    if 0 < hist_max[0] < 10: #红色
        res_color = "red"
    elif 100 < hist_max[0] < 124: #蓝色
        res_color = "blue"
    elif 35 < hist_max[0] < 85: #绿色
        res_color = "green"
    else:
        res_color = "unknown"

    return res_color

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

# 边缘检测 canny边缘检测 第一个参数：输入图像 第二个参数：低阈值 第三个参数：高阈值
edges = cv2.Canny(filtered, 30, 200)

# 寻找轮廓 CHAIN_APPROX_SIMPLE保留轮廓的端点信息
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 找到前10个contourArea轮廓面积和排序
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

# 画轮廓
cv2.drawContours(img, contours, -1, [0, 255, 0], 2)

for c in contours:
    # 排除超大的区域的轮廓， contourArea轮廓面积
    if cv2.contourArea(c) > 1024 * 768 * 0.5:
        continue

    #计算周长， True 轮廓是闭合
    peri = cv2.arcLength(c, True)

    # 多边形近似处理 控制逼近精度
    approx = cv2.approxPolyDP(c, 0.018*peri, True)

    # 如果逼近后4个点，很有可能是矩形轮廓
    if len(approx) == 4:

        #最小的外接矩形
        x, y, w, h = cv2.boundingRect(approx)
        crop_imgage = img[y : y + h, x : x + w]
        if "blue" == reg_area_color(crop_imgage):
            screenCnt = approx
            break
        elif "green" == reg_area_color(crop_imgage):
            screenCnt = approx
            break

if screenCnt is not None:
    cv2.drawContours(img, [screenCnt], -1, [0, 0, 255], 2)

#遮罩
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [screenCnt], 0, 255, -1)


# 图像位运算
cv2.bitwise_and(img, img, mask = mask)

# 图像裁剪
(x, y) = np.where(mask == 255)

# 左上方 最左边
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))

cropped = img[topx:bottomx, topy:bottomy]

#创建文本ocr model_name模型文字
model = TexRecognition(model_name = "PP-OCRv5_server_rec")
#文本识别
output = model.predict(input = " general_ocr_rec_001.png", batch_size = 1)
for res in output:
    res.print()
    res.save_to_img(save_path = "./output/")
    res.save_to_json(save_path = "./output/res/json")

show_image("cropped", cropped)
