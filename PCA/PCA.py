# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/10/18 13:25'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征
def meanX(dataX):
    return np.mean(dataX, axis=0)


def pca(XMat, k):
    np.set_printoptions(suppress=True)
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    # 扩充2x1的矩阵为150x2
    avgs = np.tile(average, (m, 1))
    # 进行均值化
    data_adjust = XMat - avgs
    # m默认一行为样本进行处理
    covX = np.cov(data_adjust.T)
    # 求取特征值和特征向量
    featValue, featVec = np.linalg.eig(covX)
    # 获得由大到小的特征值索引
    index = np.argsort(-featValue)
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        # 注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]])
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData


def loaddata(datafile):
    return np.array(pd.read_csv(datafile, sep="\t", header=-1)).astype(np.float)


def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)
    print("dataArr1=\n",dataArr1)
    print("dataArr2=\n",dataArr2)
    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_x2 = []
    axis_y1 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i, 0])
        axis_y1.append(dataArr1[i, 1])
        axis_x2.append(dataArr2[i, 0])
        axis_y2.append(dataArr2[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig("outfile.png")
    plt.show()

def main():
    datafile = "data.txt"
    XMat = loaddata(datafile)
    k = 2
    return pca(XMat,k)

if __name__ == '__main__':
    finalData,reconMat = main()
    plotBestFit(finalData,reconMat)
    from sklearn.model_selection import cross_val_score