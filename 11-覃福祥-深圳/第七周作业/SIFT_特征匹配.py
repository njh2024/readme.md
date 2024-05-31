import cv2
import numpy as np

'''
outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags)
参数：
img1：图像1
keypoints1：图像1的特征点
img2：图像2
keypoints1：图像2的特征点
matches1to2：图像1特征点到图像2特征点的匹配成功的点，keypoints1[i]和keypoints2[matches[i]]为匹配点
outImg: 绘制完的输出图像
matchColor:匹配特征点和其连线的颜色，-1时表示颜色随机
singlePointColor:未匹配点的颜色，-1时表示颜色随机
matchesMask: mask决定那些匹配点被画出，若为空，则画出所有匹配点
flags：绘图功能的标识设置
有以下4种：
1. cv2.DRAW_MATCHES_FLAGS_DEFAULT：只绘制特征点的坐标点,显示在图像上就是一个个小圆点,每个小圆点的圆心坐标都是特征点的坐标
2. cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：绘制特征点的时候绘制的是一个个带有方向的圆,这种方法同时显示图像的坐标,size，和方向,是最能显示特征的一种绘制方式
3. cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：函数不创建输出的图像,而是直接在输出图像变量空间绘制,要求本身输出图像变量就是一个初始化好了的,size与type都是已经初始化好的变量
4. cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS：单点的特征点不被绘制
'''
img1_gray = cv2.imread("iphone1.png")  # 读图
img2_gray = cv2.imread("iphone2.png")

# sift = cv2.SIFT()
sift = cv2.SIFT_create()  # 实例化SIFT函数
# sift = cv2.SURF()

kp1, des1 = sift.detectAndCompute(img1_gray, None)  # 调用SIFT函数
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# BFmatcher with default parms
bf = cv2.BFMatcher(cv2.NORM_L2)  # 实例化欧氏距离函数
# opencv中knnMatch是一种蛮力匹配，穷举匹配
# 将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
matches = bf.knnMatch(des1, des2, k=2)  # 特征匹配：调用欧式距离函数

goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:  # 筛选出了部分匹配点画图：如果相似度最高的两个特征差值大于0.5倍，则选取
        goodMatch.append(m)
goodMatch = sorted(goodMatch, key = lambda x:x.distance)
print(goodMatch)
h1, w1 = img1_gray.shape[:2]           #获取两张图像的高度和宽度
h2, w2 = img2_gray.shape[:2]

out_img1 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)     #创建一个空白图像 vis，大小为两张输入图像高度的最大值和它们宽度之和，通道数为 3（彩色图像）。
out_img1[:h1, :w1] = img1_gray        # 将两张输入的灰度图像放置在 vis 中，分别放在左右两侧
out_img1[:h2, w1:w1 + w2] = img2_gray

out_img2 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
out_img2[:h1, :w1] = img1_gray
out_img2[:h2, w1:w1 + w2] = img2_gray

out_img1 = cv2.drawMatchesKnn(img1_gray, kp1, img2_gray, kp2, matches, out_img1)        #画出所有匹配点
out_img2 = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, goodMatch[:20], out_img2,(255,255,0),None,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)   #画出筛选的部分匹配点
cv2.imshow("out_img1", out_img1)
cv2.imshow("out_img2", out_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
