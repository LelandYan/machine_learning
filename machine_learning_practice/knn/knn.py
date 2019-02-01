# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/12/3 22:49'

import numpy as np
import operator
import collections
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def classify1(intX, dataSet, labels, k):
    dist = np.sum((intX - dataSet) ** 2, axis=1) ** 0.5
    k_labels = [labels[index] for index in dist.argsort()[0:k]]
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label


def test1():
    group, labels = createDataSet()
    print(classify0([0.1, 0.1], group, labels, 3))


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    with open(filename) as f:
        index = 0
        for line in f.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
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


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    figure(datingDataMat,datingLabels)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)

    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(f"the classifier came back with {classifierResult},the real answer is {datingLabels[i]}")
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print(f"the total error rate is {errorCount / float(numTestVecs)}")
    print("numTestVecs=", numTestVecs," errorCount=",errorCount)


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


if __name__ == '__main__':
    datingClassTest()

