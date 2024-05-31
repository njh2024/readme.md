###cluster.py
#导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
linkage 函数负责生成层次信息，而 fcluster 函数负责根据这些信息进行实际的聚类。
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”,限制显示的分类数，前面的运算是穷举所有分类结果，这一步才是我们想要的分类数
'''
X=[[1,2],[3,2],[4,4],[1,2],[1,3]]
Z=linkage(X,'ward')  #生成计算过程矩阵，记录了层次聚类的层次信息，即每一步合并的簇和它们的距离。'ward'：欧氏距离法算距离
print(Z)
f=fcluster(Z,4,'distance')  #实现聚类
fig=plt.figure(figsize=(5,3))  #5表示图片的长,3表示图片的宽,单位是inch
dn=dendrogram(Z)   # 制作谱系图
plt.show()




