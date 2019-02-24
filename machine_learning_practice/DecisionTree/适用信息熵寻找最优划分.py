# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/24 13:34'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(1, -1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(1, -1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmp = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmp)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:,2:]
    y = iris.target

    dt_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
    model = dt_clf.fit(X, y)
    plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.scatter(X[y == 2, 0], X[y == 2, 1])
    plt.show()
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    # n_estimators 集成几个模型
    # max_samples 模型看几个数据
    # bootstrap 是否放回取样
    bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                    n_estimators=500,max_samples=100,
                                    bootstrap=True)
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import RandomForestRegressor