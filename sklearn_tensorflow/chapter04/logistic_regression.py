# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/11 19:11'

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris.DESCR)
# print(iris['data'].shape)
# X = iris["data"][:,3:]
# y = (iris["target"] == 2).astype(np.int)
#
from sklearn.linear_model import LogisticRegression

# log_reg = LogisticRegression(solver="liblinear",random_state=42)
# log_reg.fit(X,y)
#
# X_new = np.linspace(0,3,1000).reshape(-1,1)
# y_proba = log_reg.predict_proba(X_new)
# plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
# plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
# plt.legend()
# plt.show()

# X = iris["data"][:, (2, 3)]  # petal length, petal width
# y = iris["target"]

# softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
# softmax_reg.fit(X, y)
# x0, x1 = np.meshgrid(
#         np.linspace(0, 8, 500).reshape(-1, 1),
#         np.linspace(0, 3.5, 200).reshape(-1, 1),
#     )
# X_new = np.c_[x0.ravel(), x1.ravel()]
#
#
# y_proba = softmax_reg.predict_proba(X_new)
# y_predict = softmax_reg.predict(X_new)
#
# zz1 = y_proba[:, 1].reshape(x0.shape)
# zz = y_predict.reshape(x0.shape)
#
# plt.figure(figsize=(10, 4))
# plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
# plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolor")
# plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")
#
# from matplotlib.colors import ListedColormap
# custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
#
# plt.contourf(x0, x1, zz, cmap=custom_cmap)
# contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
# plt.clabel(contour, inline=1, fontsize=12)
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(loc="center left", fontsize=14)
# plt.axis([0, 7, 0, 3.5])
# # save_fig("softmax_regression_contour_plot")
# plt.show()

#
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]


X_with_bias = np.c_[np.ones([len(X), 1]), X]
np.random.seed(2048)

test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:train_size + test_size]]
y_valid = y[rnd_indices[train_size:train_size + test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]


def to_one_hot(y):
    n_classes = y.max() + 1
    m = len(y)
    Y_one_hot = np.zeros((m,n_classes))
    Y_one_hot[np.arange(m),y] = 1
    return Y_one_hot

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)

def softmax(logits):
    exps = np.exp(logits)
    exps_sums = np.sum(exps,axis=1,keepdims=True)
    return exps / exps_sums

n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))

eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

Theta = np.random.randn(n_inputs,n_outputs)
for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon),axis=1))
    error = Y_proba - Y_train_one_hot
    if iteration % 500 == 0:
        print(iteration,loss)
    gradient = 1 / m * X_train.T.dot(error)
    Theta = Theta - eta * gradient

