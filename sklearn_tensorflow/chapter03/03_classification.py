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

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/7,random_state=42)

# Binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000,tol=1e-3,random_state=42)
sgd_clf.fit(X_train,y_train_5)
some_digit = X[36000]
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

y_scores = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method='decision_function')
# print(y_scores.shape)
# print(y_scores)
from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds = precision_recall_curve(y_train_5,y_scores[:,1])

def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds,precisions[:-1],'b--',label='Precision',linewidth=2)
    plt.plot(thresholds,recalls[:-1],'g-',label='Recall',linewidth=2)
    plt.legend(loc='center right',fontsize=16)
    plt.xlabel('Threshold',fontsize=10)
    plt.grid(True)
    plt.axis([-60000,60000,0,1])

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

def shift_image(image,dx,dy):
    image = image.reshape((28,28))
    shift_image = shift(image,[dx,dy],cval=0,mode='constant')
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
from sklearn.preprocessing import Imputer,LabelBinarizer

from sklearn.datasets import load_iris

