import cv2
import numpy as np
'''
img = cv2.drawKeypoints(image, keypoints, outImage, color, flags)
参数：
image：输入的灰度图像
keypoints：从原图中获得的关键点，这也是画图时所用到的数据
outImage：输出的图片
color：绘制关键点所用颜色
flags：绘图功能的标识设置
有以下4种：
1. cv2.DRAW_MATCHES_FLAGS_DEFAULT：只绘制特征点的坐标点,显示在图像上就是一个个小圆点,每个小圆点的圆心坐标都是特征点的坐标
2. cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：绘制特征点的时候绘制的是一个个带有方向的圆,这种方法同时显示图像的坐标,size，和方向,是最能显示特征的一种绘制方式
3. cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：函数不创建输出的图像,而是直接在输出图像变量空间绘制,要求本身输出图像变量就是一个初始化好了的,size与type都是已经初始化好的变量
4. cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS：单点的特征点不被绘制
'''

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()  #opencv4.5版本可以直接调用SIFT了，实例化SIFT
keypoints,descriptor=sift.detectAndCompute(gray,None)    #找出关键点并计算描述符(是对关键点周围区域的描述，通常是一组数值或向量)，detectAndCompute：该函数可以同时执行检测与计算
#print(descriptor)
#画出关键点
img=cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                      color=(51,163,236))
cv2.imshow('sift_keypoints',img)
cv2.waitKey()
cv2.destroyAllWindows()
