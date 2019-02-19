# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/19 15:20'
import numpy as np


def train_test_split(X, y, test_size=0.2, seed=None):
    """将数据X和y按照test_ratio分割成X_train X_test y_train y_test"""

    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_size <= 1.0, \
        "test_ratio must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_size)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test