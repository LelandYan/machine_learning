# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/23 22:36'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target == 9] = 1
y[digits.target != 9] = 0

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
# scores = log_reg.score(X_test,y_test)
# print(scores)
y_log_predict = log_reg.predict(X_test)

def TN(y_true,y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

def FP(y_true,y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

def FN(y_true,y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

def TP(y_true,y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

def confusion_matrix(y_true,y_predict):
    return np.array([
        [TN(y_true,y_predict),FP(y_true,y_predict)],
        [FN(y_true,y_predict),TP(y_true,y_predict)]
    ])

print(confusion_matrix(y_test,y_log_predict))

def precision_score(y_true,y_predict):
    tp = TP(y_true,y_predict)
    fp = FP(y_true,y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0

def recall_score(y_true,y_predict):
    tp = TP(y_true,y_predict)
    fn = FN(y_true,y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
