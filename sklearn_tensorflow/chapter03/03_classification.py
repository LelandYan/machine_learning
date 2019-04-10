# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/5 23:21'
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 导入数据
mnist = loadmat("mnist-original.mat")
X = mnist["data"].T
y = mnist["label"][0]

# some_digit = X[0]
# some_digit_image = some_digit.reshape(28,28)
# plt.imshow(some_digit_image,cmap=mpl.cm.binary,interpolation="nearest")
# plt.axis('off')
# plt.show()
# print(y[0])

y = y.astype(np.uint8)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary)
    plt.axis('off')


# # EXTRA
def plot_digits(instances, image_per_row=10, **options):
    size = 28
    # 一行显示有多少张图片
    image_per_row = min(len(instances), image_per_row)
    # 将图片变为28*28形状
    images = [instance.reshape(size, size) for instance in instances]

    n_rows = (len(instances) - 1) // image_per_row + 1
    row_images = []
    # 剩余还有几个空余的位置
    n_empty = n_rows * image_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * image_per_row:(row + 1) * image_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


# plt.figure(figsize=(9, 9))
# example_images = X[:100]
# plot_digits(example_images, image_per_row=10)
# plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 7, random_state=42)

# Binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

# sgd_clf = SGDClassifier(max_iter=1000,tol=1e-3,random_state=42)
# sgd_clf.fit(X_train,y_train_5)
# some_digit = X[36000]
# print(y[36000])
# print(sgd_clf.predict([some_digit]))
# plot_digit(X[36000])
# plt.show()


# from sklearn.model_selection import cross_val_score
# res = cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring='accuracy')
# print(res)
#
# from sklearn.model_selection import StratifiedKFold
# from sklearn.base import clone
#
# skfolds = StratifiedKFold(n_splits=3,random_state=42)
#
# for train_index,test_index in skfolds.split(X_train,y_train_5):
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = y_train_5[train_index]
#     X_test_fold = X_train[test_index]
#     y_test_fold = y_train_5[test_index]
#
#     clone_clf.fit(X_train_folds,y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct  =sum(y_pred == y_test_fold)
#     print(n_correct / len(y_pred))

from sklearn.model_selection import cross_val_predict

# y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_train_5,y_train_pred))

# y_scores = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method='decision_function')
# print(y_scores.shape)
# print(y_scores)
from sklearn.metrics import precision_recall_curve


# precisions,recalls,thresholds = precision_recall_curve(y_train_5,y_scores[:,1])

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
    plt.legend(loc='center right', fontsize=16)
    plt.xlabel('Threshold', fontsize=10)
    plt.grid(True)
    plt.axis([-60000, 60000, 0, 1])


# plt.figure(figsize=(8,4))
# plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
# plt.plot([7813, 7813], [0., 0.9], "r:")         # Not shown
# plt.plot([-50000, 7813], [0.9, 0.9], "r:")      # Not shown
# plt.plot([-50000, 7813], [0.4368, 0.4368], "r:")# Not shown
# plt.plot([7813], [0.9], "ro")                   # Not shown
# plt.plot([7813], [0.4368], "ro")
# plt.show()
# from sklearn.metrics import roc_curve
# from sklearn.dummy import DummyClassifier
# dmy_clf = DummyClassifier()
# y_probas_dmy = cross_val_predict(dmy_clf,X_train,y_train_5,cv=3,method='predict_proba')
# y_scores_dmy = y_probas_dmy[:,1]
# fpr,tpr,thresholds = roc_curve(y_train_5,y_scores_dmy)

from sklearn.neighbors import KNeighborsClassifier
# knn_clf = KNeighborsClassifier(weights='distance',n_neighbors=4)
# knn_clf.fit(X_train,y_train)
# y_knn_pred = knn_clf.predict(X_test)
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,y_knn_pred))

from scipy.ndimage.interpolation import shift


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shift_image = shift(image, [dx, dy], cval=0, mode='constant')
    return shift_image.reshape([-1])


# image = X_train[1000]
# shift_image_down = shift_image(image,0,5)
# shift_image_left = shift_image(image,-5,0)
#
# plt.figure(figsize=(12,3))
# plt.subplot(131)
# plt.title('Original',fontsize=14)
# plt.imshow(image.reshape(28,28),interpolation='nearest',cmap='Greys')
# plt.subplot(132)
# plt.title("Shifted down", fontsize=14)
# plt.imshow(shift_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
# plt.subplot(133)
# plt.title("Shifted left", fontsize=14)
# plt.imshow(shift_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
# plt.show()
from sklearn.preprocessing import Imputer, LabelBinarizer

from sklearn.datasets import load_iris

iris = load_iris()
from sklearn.preprocessing import FunctionTransformer
from numpy import log1p
# FunctionTransformer(log1p).fit_transform(iris.data)


from sklearn.feature_selection import VarianceThreshold

print(iris.data.shape)
print(iris.target.shape)
# data = VarianceThreshold(threshold=3).fit_transform(iris.data)
# print(data.shape)


# 相关系数法
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

X = iris.data
# print(iris.data.T.shape)

# data = SelectKBest(lambda X, Y: np.array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
# print(data.shape)
from sklearn.feature_selection import chi2, SelectKBest
# 选择K个最好的特征，返回选择特征后的数据
# res = SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
# print(res.shape)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# re =RFE(estimator=LogisticRegression(),n_features_to_select=2).fit_transform(iris.data,iris.target)
# print(re.shape)

from sklearn.feature_selection import SelectFromModel

# res = SelectFromModel(LogisticRegression(penalty='l2',C=0.1)).fit_transform(iris.data,iris.target)
# print(res.shape)


from sklearn.linear_model import LogisticRegression


class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        # 权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight,
                                    random_state=random_state, solver=solver, max_iter=max_iter,
                                    multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        # 使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                     intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver, max_iter=max_iter,
                                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        # 训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        # 训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        # 权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                # L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    # 对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        # 在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    # 计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self


from sklearn.feature_selection import SelectFromModel

# 带L1和L2惩罚项的逻辑回归作为基模型的特征选择
# 参数threshold为权值系数之差的阈值
res = SelectFromModel(LR(threshold=0.5, C=0.2)).fit_transform(iris.data, iris.target)
print(res == iris.data)
