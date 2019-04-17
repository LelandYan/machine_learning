# _*_ coding: utf-8 _*_

from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris['data'][:,(2,3)]
y = iris['target']

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# SVM Classifier model
svm_clf = SVC(kernel='linear',C=float('inf'))
svm_clf.fit(X,y)
# print(svm_clf)

# Bad models
x0 = np.linspace(0,5.5,200)
pred_1 = 5 * x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5

def plot_svc_decision_boundary(svm_clf,xmin,xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]