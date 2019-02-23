# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/22 11:44'

import numpy as np


class PCA:
    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None

    def __repr__(self):
        return f"PCA(n_components={self.n_components})"

    def fit(self, X, eta=0.01, n_iter=1e4):
        """获取数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_components must be greater than the feature number of X"

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

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta)
            self.components_[i, :] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分上"""
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """给定X，反向映射会原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)


if __name__ == '__main__':
    # X = np.empty((100, 2))
    # np.random.seed(666)
    # X[:, 0] = np.random.uniform(0., 100, size=100)
    # X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)
    # pca = PCA(n_components=1)
    # pca.fit(X)
    # # print(pca.components_)
    # X_reduction = pca.transform(X)
    # print(X_reduction.shape)
    # X_restore = pca.inverse_transform(X_reduction)
    # print(X_restore.shape)
    from sklearn.decomposition import PCA
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    from sklearn.neighbors import KNeighborsClassifier

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    score1 = knn_clf.score(X_test, y_test)
    print(score1)

    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_train_reduction = pca.transform(X_train)
    X_test_reduction = pca.transform(X_test)

    knn_clf2 = KNeighborsClassifier()
    knn_clf2.fit(X_train_reduction, y_train)
    score2 = knn_clf2.score(X_test_reduction, y_test)
    print(pca.explained_variance_ratio_)
    # print(score2)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_reduction = pca.transform(X)
    for i in range(10):
        plt.scatter(X_reduction[y == i, 0], X_reduction[y == i, 1], alpha=0.8)
    plt.show()
