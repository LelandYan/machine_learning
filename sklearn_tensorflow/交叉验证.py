# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/7 22:58'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm

iris = load_iris()

# print(iris.data.shape,iris.target.shape)

# train_test_split 对数据集进行快速打乱
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.4,random_state=42)

# clf = svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
# print(clf.score(X_test,y_test))


# cross_val_score 对数据集进行指定次数的交叉验证并为每次的验证结果评测
from sklearn.model_selection import cross_val_score

clf  = svm.SVC(kernel='linear',C=1)
scores = cross_val_score(clf,iris.data,iris.target,cv=5)
# print(scores)
# print(scores.mean())

# cross_val_predict 返回的是estimator的分类的结果（或者回归值）
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf,iris.data,iris.target,cv=10)
# print(predicted)
from sklearn.metrics import accuracy_score
# print(accuracy_score(iris.target,predicted))

# KFold 每次放回不会有重叠，相当于无放回抽样
from sklearn.model_selection import KFold
X = ['a','b','c','d']
kf = KFold(n_splits=2)

# for train,test in kf.split(X):
#     print(train)
#     print(test)
#     print(np.array(X)[train],np.array(X)[test])
#

# ShuffleSplit 是有放回的抽样
from sklearn.model_selection import ShuffleSplit
# X = np.arange(5)
# ss = ShuffleSplit(n_splits=3,test_size=0.2,random_state=42)
# for train_index,test_index in ss.split(X):
#     print(train_index,test_index)
#


# StratifiedKFold 指定分组进行无放回抽样
from sklearn.model_selection import StratifiedKFold


a = np.array([-1.3828788, -1.4302242])

b = np.array([9,66])

from sklearn.metrics import precision_recall_curve,auc
precisoin_r,recall_r,thresholds_r = precision_recall_curve(a,b)


