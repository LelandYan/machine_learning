# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/24 10:02'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y < 2, :2]
    y = y[y < 2]
    standardScaler = StandardScaler()
    standardScaler.fit(X)
    X_standard = standardScaler.transform(X)
    svc = LinearSVC(C=1e9)
    print(svc.fit(X_standard,y))
    from sklearn.svm import SVC

    # plt.scatter(X[y==0,0],X[y==0,1])
    # plt.scatter(X[y==1,0],X[y==1,1])
    # plt.show()
    from sklearn.ensemble import GradientBoostingClassifier
    gb_clf = GradientBoostingClassifier(max_depth=2,n_estimators=20)
    
