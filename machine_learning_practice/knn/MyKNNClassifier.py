# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/19 13:24'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from collections import Counter
from machine_learning_practice.knn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric

class KNNClassifier:
    def __init__(self, k=3):
        """初始化KNN分类器"""
        assert k >= 1, "k must be valid"
        self._k = k
        self._X_train = None
        self._Y_train = None

    def fit(self, X_train, Y_train):
        """根据训练数据集X_train和Y_train训练KNN分类器"""
        assert X_train.shape[0] == Y_train.shape[0], \
            "the size of X_train must be equal to the size of Y_train"
        assert self._k <= X_train.shape[0], \
            "the size of X_train must be at least k"
        self._X_train = X_train
        self._Y_train = Y_train
        return self

    def predict(self, X_predict):
        """给定预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._Y_train is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测的数据X，返回x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        distance = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distance)
        topK_y = [self._Y_train[i] for i in nearest[:self._k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return f"KNN(k={self._k})"


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    my_knn_clf = KNNClassifier(3)
    my_knn_clf.fit(X_train, y_train)
    y_predict = my_knn_clf.predict(X_test)
    print(np.sum(y_predict == y_test)/ len(y_test))
    # y = y.reshape(150,-1)
    # data = np.hstack((X,y))
