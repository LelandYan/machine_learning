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
    retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == index and v == value]
    return retDataSet