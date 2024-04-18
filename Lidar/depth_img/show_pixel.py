import cv2
import numpy as np

# 保存图片的初始大小
id = 12
original_img = cv2.imread(rf'G:\LangYa\Radar\code\Radar_Develop\images\img{id}.bmp')
img = original_img.copy()
# 分割出图片的名称，不要后缀
print(id,":")
# 添加一个全局变量来保存放大或缩小的比例
scale = 1.0

def show_pixel(event, x, y, flags, param):
    global img, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        # 将坐标转换回原始图像的坐标
        original_x = int(x / scale)
        original_y = int(y / scale)
        print(f"Original coordinate: x: {original_x}, y: {original_y}")
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0x780000:  # 向上滚动
            img = cv2.resize(img, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_LINEAR)
            scale *= 1.1
        else:  # 向下滚动
            img = cv2.resize(img, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_LINEAR)
            scale *= 0.9
        cv2.imshow('image', img)

cv2.namedWindow('image')
cv2.setMouseCallback('image', show_pixel)
cv2.imshow('image', img)
cv2.waitKey(0)