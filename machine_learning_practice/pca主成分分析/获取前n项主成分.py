# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/21 19:58'

import numpy as np
import matplotlib.pyplot as plt


def demean(X):
    return X - np.mean(X, axis=0)


def f(w, X):
    return np.sum((X.dot(w) ** 2)) / len(X)


def df(w, X):
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


def first_component(X, initial_w, eta, n_iter=1e4, epsilon=1e-8):
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


def first_n_components(n, X, eta=0.01, n_iter=1e4, epsilon=1e-8):
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res = []
    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = first_component(X_pca, initial_w, eta)
        res.append(w)

        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

    return res


if __name__ == '__main__':
    X = np.empty((100, 2))
    np.random.seed(666)
    X[:, 0] = np.random.uniform(0., 100, size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)
    X = demean(X)
    print(first_n_components(2,X))
    # plt.scatter(X[:, 0], X[:, 1])
    # initial_w = np.random.random(X.shape[1])
    # eta = 0.01
    # w = first_component(X, initial_w, eta)
    # # 向量化
    # # X2 = np.empty(X.shape)
    # X2 = X - X.dot(w).reshape(-1, 1) * w.reshape(1, -1)
    # # for i in range(len(X)):
    # #     X2[i] = X[i] - X[i].dot(w) * w
    # # plt.scatter(X2[:, 0], X2[:, 1])
    # # plt.show()
    # w2 = first_component(X2, initial_w, eta)
    # # print(w2.dot(w))
