# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/12/27 18:36'

import numpy as np
import matplotlib.pyplot as plt
import operator
import os
from sklearn.neighbors import KNeighborsClassifier
def classify0(intX, dataSet, labels, k):
    '''
    :param intX:用于分类的输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return:
    '''
    # 去样本的数量
    dataSetSize = dataSet.shape[0]
    # 计算欧拉距离
    diffMat = np.tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 根据距离排序从小到大排列，并提取对应的index，输出
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = (dataSet - minVals) / ranges
    return normDataSet, ranges, minVals


def figure(dataingDataMat, datingLabels):
    fig = plt.figure(figsize=(10, 10))
    plt.title("Example:improving matching from a dating site with KNN")
    ax1 = fig.add_subplot(311)
    ax1.scatter(x=dataingDataMat[:, 0], y=dataingDataMat[:, 1], s=15.0 * np.array(datingLabels),
                c=np.array(datingLabels))
    # ax1.set_xlabel("Percentage Time Spent Playing Video Games")
    # ax1.set_ylabel("Frequent FlyIer Miles Earned Per Year")
    ax2 = fig.add_subplot(312)
    ax2.scatter(x=dataingDataMat[:, 1], y=dataingDataMat[:, 2], s=15.0 * np.array(datingLabels),
                c=np.array(datingLabels))
    ax3 = fig.add_subplot(313)
    ax3.scatter(x=dataingDataMat[:, 0], y=dataingDataMat[:, 2], s=15.0 * np.array(datingLabels),
                c=np.array(datingLabels))
    plt.show()


def datingClassesTest():
    hoRatio = 0.1
    dataingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    figure(dataingDataMat,datingLabels)
    normMat, ranges, minVals = autoNorm(dataingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

def img2vector(filename):
    '''
    :param filename:
    :return:
    '''
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(f"trainingDigits/{fileNameStr}")

    testFileList = os.listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(f"testDigits/{fileNameStr}")
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print(f"the classifier came back with: {classifierResult},the real answer is:{classNumStr}")
        if (classifierResult != classNumStr):errorCount += 1.0
    print(f"\nthe total number of errors is : {errorCount}")
    print(f"\nthe total error rate is {errorCount / float(mTest)}")

if __name__ == '__main__':
    datingClassesTest()
    handwritingClassTest()