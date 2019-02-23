# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/23 18:56'

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


if __name__ == '__main__':
    from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    x = np.linspace(-10,10,500)
    y = sigmoid(x)
    plt.plot(x,y)
    plt.show()