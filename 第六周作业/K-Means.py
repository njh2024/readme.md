# coding: utf-8

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签，在具体实现时，为了获得最佳分类效果，可能需要使用不同的初始分类值进
            行多次尝试。指定 attempts 的值，可以让算法使用不同的初始值进行多次（attempts 次）尝试。
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示每个分类点的中心点数据，输出（M,N）矩阵数据，每一行代表一类，每一列代表不同通道
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img=cv2.imread('lenna.png',0)
print(img.shape)
rows,cols=img.shape[:2]  #获取图像高度、宽度

#data=img.reshape((rows*cols,1))#图像二维像素转换为一维，reshape(1,-1)转化成1行，列需要计算：reshape(2,-1)转换成两行，列需要计算：reshape(-1,1)转换成1列，行需要计算：reshape(-1,2)转化成两列，行需要计算。
data=img.reshape((-1,1))
data=np.float32(data)
#print('data:',data)
print('data:',data.shape)

#停止条件 (type,max_iter,epsilon)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#设置标签
flags=cv2.KMEANS_RANDOM_CENTERS    #随机选择初始质心

#K-Means聚类 聚集成4类
compactness,labels,centers=cv2.kmeans(data,4,None,criteria,10,flags)   #重复试验kmeans算法的次数，flags随机生成初始质心，所以相当于尝试10次不同初始质心聚类的结果，返回最佳结果标签
print('labels:',labels.shape)
print('centers:',centers.shape)
#生成最终图像
dst=labels.reshape((img.shape[0],img.shape[1]))    #一维再转化为二维图像显示,因为一开始data聚合成一维是是按照行>列像素的顺序，此时labels再分裂也是行>列的顺序，图像像素位置不变

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles=[u'原始图像',u'聚类图像']
images=[img,dst]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i],'gray')
    plt.xticks([])
    plt.yticks([])
plt.show()