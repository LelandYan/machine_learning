# _*_ coding: utf-8 _*_
from sklearn.datasets import fetch_mldata
from scipy.sparse import lil_matrix
import numpy as np
from scipy.io import loadmat

# 导入数据
mnist = loadmat("mnist-original.mat")
X = mnist["data"].T
y = mnist["label"][0]

import matplotlib.pyplot as plt
import matplotlib

some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
# plt.show()

# 划分数据集
X_train,X_test,y_train,y_test = X[:60000],X[60000:],y[:60000],y[60000:]

# 打乱数据集，保证交叉验证的每一折都是相似的
shuffle_index = np.random.permutation(60000)
X_train,y_train = X_train[shuffle_index],y_train[shuffle_index]


# 训练一个二分类器
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

"""
现在让我们挑选一个分类器去训练它。用随机梯度下降分类器 SGD，是一个不错的开始。使用 Scikit-Learn 的SGDClassifier类。
这个分类器有一个好处是能够高效地处理非常大的数据集。这部分原因在于SGD一次只处理一条数据，
这也使得 SGD 适合在线学习（online learning）。
我们在稍后会看到它。让我们创建一个SGDClassifier和在整个数据集上训练它。"""

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)

print(sgd_clf.predict([some_digit]))

# 使用交叉验证来测量准确性
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
skfolds = StratifiedKFold(n_splits=3,random_state=42)
for train_index,test_index in skfolds.split(X_train,y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct/len(y_pred))

# 用sklearn简单实现
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy"))

# 猜测非5的准确率
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self,X,y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)
never_5_clf = Never5Classifier()
res = cross_val_score(never_5_clf,X_train,y_train_5,cv=3,scoring="accuracy")
print(res)

# 在预测是否不是5的分类器中，对于有偏差的数据集来说，准确率并不是一个好的性能评价指标

# cross_val_score 是求取交叉验证的返回的正确率的
# cross_val_predict 是采用测试集进行计算的,它不是返回一个评估分数，而是返回基于每一个测试折做出的一个预测值
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
# 混淆矩阵
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5,y_train_pred)

# recall precision
from sklearn.metrics import precision_score,recall_score
precision_score(y_train_5,y_train_pred)
recall_score(y_train_5,y_train_pred)

# F1
"""
通常结合准确率和召回率会更加方便，这个指标叫做“F1 值”，特别是当你需要一个简单的方法去比较两个分类器的优劣的时候。
F1 值是准确率和召回率的调和平均。普通的平均值平等地看待所有的值，而调和平均会给小的值更大的权重。
所以，要想分类器得到一个高的 F1 值，需要召回率和准确率同时高。"""
from sklearn.metrics import f1_score
f1_score(y_train_5,y_train_pred)
