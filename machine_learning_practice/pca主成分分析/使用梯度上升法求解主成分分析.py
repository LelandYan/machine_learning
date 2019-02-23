# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/21 11:12'

import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:, 0] = np.random.uniform(0., 100, size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)


# plt.scatter(X[:, 0], X[:, 1])


def demean(X):
    return X - np.mean(X, axis=0)


X_demean = demean(X)
plt.scatter(X_demean[:, 0], X_demean[:, 1])
# plt.show()


def f(w, X):
    return np.sum((X.dot(w) ** 2)) / len(X)


def df_math(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)


def df_bug(w, X, epsilon=0.0001):
    res = np.empty(len(w))
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
    return res


def direction(w):
    return w / np.linalg.norm(w)


def gradient_ascent(df, X, initial_w, eta, n_iter=1e4, epsilon=1e-8):
    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iter:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)
        if (abs(f(w, X) - f(last_w, X)) < epsilon):
            break

        cur_iter += 1
    return w

initial_w = np.random.random(X.shape[1])
eta = 0.01
re1 = gradient_ascent(df_bug,X_demean,initial_w,eta)
re2 = gradient_ascent(df_math,X_demean,initial_w,eta)
print(re1,re2)
