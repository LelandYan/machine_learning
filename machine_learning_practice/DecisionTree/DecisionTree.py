# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/12/30 23:01'

import operator
from math import log
from collections import Counter


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 1, 'no'],
        [1, 1, 'no'],
        [1, 1, 'no'],
    ]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):
    label_count = Counter(data[-1] for data in dataSet)
    probs = [p / len(dataSet) for p in  label_count.values()]
    shannonEnt = sum([-p * log(p,2) for p in probs])
    return shannonEnt

def splitDataSet(dataSet,index,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            retDataSet.append(reducedFeatVec)
    # retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == index and v == value]
    return retDataSet

if __name__ == '__main__':
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 1, 'no'],
        [1, 1, 'no'],
        [1, 1, 'no'],
    ]
    label_count = Counter(data[-1] for data in dataSet)
    probs = [p / len(dataSet) for p in label_count.values()]
    shannonEnt = sum([-p * log(p, 2) for p in probs])
# if __name__ == '__main__':
#     dataSet = [
#         [1, 1, 'yes'],
#         [1, 1, 'yes'],
#         [1, 1, 'no'],
#         [1, 1, 'no'],
#         [1, 1, 'no'],
#     ]
#     index = 1
#     value = 1
#     retDataSet = []
#     for featVec in dataSet:
#         # index列为value的数据集【该数据集需要排除index列】
#         # 判断index列的值是否为value
#         if featVec[index] == value:
#             # chop out index used for splitting
#             # [:index]表示前index行，即若 index 为2，就是取 featVec 的前 index 行
#             reducedFeatVec = featVec[:index]
#             reducedFeatVec.extend(featVec[index + 1:])
#             # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
#             # 收集结果值 index列为value的行【该行需要排除index列】
#             retDataSet.append(reducedFeatVec)
#     print(retDataSet)
#     retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == index and v == value]
#     print(retDataSet)