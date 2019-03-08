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
# some_digit_image = some_digit.reshape(28,28)
# plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
# plt.axis("off")
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

# print(sgd_clf.predict([some_digit]))

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
    # print(n_correct/len(y_pred))

# 用sklearn简单实现
from sklearn.model_selection import cross_val_score
# print(cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy"))

# 猜测非5的准确率
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self,X,y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)
never_5_clf = Never5Classifier()
res = cross_val_score(never_5_clf,X_train,y_train_5,cv=3,scoring="accuracy")
# print("cross_val_score",res)

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


# 准确率和召回率之间的折中
y_score = sgd_clf.decision_function([some_digit])
# print(y_score)
threshold = 0
y_some_digit_pred = (y_score > threshold)
# print(y_some_digit_pred)

# 提高阈值会降低召回率
# 返回决策的分数，而不是预测值
y_score = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")
# y_scores = sgd_clf.decision_function(X_train)
from sklearn.metrics import precision_recall_curve
precisions,recalls,threshold = precision_recall_curve(y_train_5,y_score[:,1])

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

# plot_precision_recall_vs_threshold(precisions, recalls, threshold)
# plt.show()

# 自定义阈值来求取预测的准确率和召回率
y_train_pred_90 = (y_score[:,1] > 70000)
# print(precision_score(y_train_5,y_train_pred_90))
# print(recall_score(y_train_5,y_train_pred_90))


# ROC曲线
from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(y_train_5,y_score[:,1])
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
# plot_roc_curve(fpr,tpr)
# plt.show()

# auc面积 一个比较分类器之间优劣的方法是：测量ROC曲线下的面积（AUC）。一个完美的分类器的 ROC AUC 等于 1，
# 而一个纯随机分类器的 ROC AUC 等于 0.5。Scikit-Learn 提供了一个函数来计算 ROC AUC：
from sklearn.metrics import roc_auc_score
# print(roc_auc_score(y_train_5,y_score[:,1]))

"""因为 ROC 曲线跟准确率/召回率曲线（或者叫 PR）很类似，你或许会好奇如何决定使用哪一个曲线呢？一个笨拙的规则是，
优先使用 PR 曲线当正例很少，或者当你关注假正例多于假反例的时候。其他情况使用 ROC 曲线。
举例子，回顾前面的 ROC 曲线和 ROC AUC 数值，你或许认为这个分类器很棒。但是这几乎全是因为只有少数正例（“是 5”），
而大部分是反例（“非 5”）。
相反，PR 曲线清楚显示出这个分类器还有很大的改善空间（PR 曲线应该尽可能地靠近右上角）
"""

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf,X_train,y_train_5,cv=3,method="predict_proba")
y_scores_forest = y_probas_forest[:,1]
fpr_forest,tpr_forest,thresholds_forest = roc_curve(y_train_5,y_scores_forest)
# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="bottom right")
# plt.show()

# 多分类问题
# sgd_clf.fit(X_train,y_train)
# print(sgd_clf.predict([some_digit]))
# some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)
# print(np.argmax(some_digit_scores))
# print(sgd_clf.classes_)
# print(sgd_clf.classes_[np.argmax(some_digit_scores)])

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train,y_train)
# res = ovo_clf.predict([some_digit])
# print(res)
# print(len(ovo_clf.estimators_))

# 误差分析
# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
# # y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# # conf_matrix = confusion_matrix(y_train,y_train_pred)
# # # 混淆矩阵可视化
# # plt.matshow(conf_matrix,cmap=plt.cm.gray)
# # plt.show()

# 多标签分类
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large,y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)
print(knn_clf.predict([some_digit]))

# 多输出,多类分类
