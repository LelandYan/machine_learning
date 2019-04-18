# _*_ coding: utf-8 _*_

from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#
# iris = datasets.load_iris()
# X = iris['data'][:, (2, 3)]
# y = iris['target']
#
# setosa_or_versicolor = (y == 0) | (y == 1)
# X = X[setosa_or_versicolor]
# y = y[setosa_or_versicolor]

# SVM Classifier model
# svm_clf = SVC(kernel='linear', C=float('inf'))
# svm_clf.fit(X, y)
# # print(svm_clf)
#
# # Bad models
# x0 = np.linspace(0, 5.5, 200)
# pred_1 = 5 * x0 - 20
# pred_2 = x0 - 1.8
# pred_3 = 0.1 * x0 + 0.5


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

# plt.figure(figsize=(12,2.7))
#
# plt.subplot(121)
# plt.plot(x0, pred_1, "g--", linewidth=2)
# plt.plot(x0, pred_2, "m-", linewidth=2)
# plt.plot(x0, pred_3, "r-", linewidth=2)
# plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
# plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(loc="upper left", fontsize=14)
# plt.axis([0, 5.5, 0, 2])
#
# plt.subplot(122)
# plot_svc_decision_boundary(svm_clf, 0, 5.5)
# plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
# plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")
# plt.xlabel("Petal length", fontsize=14)
# plt.axis([0, 5.5, 0, 2])
#
# plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC,SVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

# svm_clf = Pipeline([
#     ("scaler",StandardScaler()),
#     ("linear_svc",LinearSVC(C=1,loss="hinge",random_state=42))
# ])
# svm_clf.fit(X,y)
# print(svm_clf.predict([[5.5, 1.7]]))
# scaler = StandardScaler()
# svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
# svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)
#
# scaled_svm_clf1 = Pipeline([
#         ("scaler", scaler),
#         ("linear_svc", svm_clf1),
#     ])
# scaled_svm_clf2 = Pipeline([
#         ("scaler", scaler),
#         ("linear_svc", svm_clf2),
#     ])
#
# scaled_svm_clf1.fit(X, y)
# scaled_svm_clf2.fit(X, y)
#
# b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
# b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
# w1 = svm_clf1.coef_[0] / scaler.scale_
# w2 = svm_clf2.coef_[0] / scaler.scale_
# svm_clf1.intercept_ = np.array([b1])
# svm_clf2.intercept_ = np.array([b2])
# svm_clf1.coef_ = np.array([w1])
# svm_clf2.coef_ = np.array([w2])
#
# # Find support vectors
# t = y * 2 -1
# support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
# support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
# svm_clf1.support_vectors_ = X[support_vectors_idx1]
# svm_clf2.support_vectors_ = X[support_vectors_idx2]
#
# plt.figure(figsize=(12,3.2))
# plt.subplot(121)
# plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris-Virginica")
# plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris-Versicolor")
# plot_svc_decision_boundary(svm_clf1, 4, 6)
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(loc="upper left", fontsize=14)
# plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
# plt.axis([4, 6, 0.8, 2.8])
#
# plt.subplot(122)
# plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
# plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
# plot_svc_decision_boundary(svm_clf2, 4, 6)
# plt.xlabel("Petal length", fontsize=14)
# plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
# plt.axis([4, 6, 0.8, 2.8])
# plt.show()

from sklearn.datasets import make_moons
X,y = make_moons(n_samples=100,noise=0.15,random_state=42)

def plot_datasets(X,y,axes):
    plt.plot(X[:,0][y==0],X[:,1][y==0],'bs')
    plt.plot(X[:,0][y==1],X[:,1][y==1],'g^')
    plt.axis(axes)
    plt.grid(True,which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


# plot_datasets(X, y, [-1.5, 2.5, -1, 1.5])
# plt.show()
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline([
    ('poly_features',PolynomialFeatures(degree=3)),
    ('scaler',StandardScaler()),
    ('svm_clf',LinearSVC(C=10,loss='hinge',random_state=42))
])
polynomial_svm_clf.fit(X,y)
def plot_predictions(clf,axes):
    x0s = np.linspace(axes[0],axes[1],100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(),x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0,x1,y_pred,cmap=plt.cm.brg,alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

    #plt.contourf(x0,x1,)
# plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
# plot_datasets(X, y, [-1.5, 2.5, -1, 1.5])
#
# plt.show()

gamma1,gamma2 = 0.1,5
C1,C2 = 0.001,1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clf = []
for gamma,C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
        ('scaler',StandardScaler()),
        ('svm_clf',SVC(kernel='rbf',gamma=gamma,C=C))
    ])
    rbf_kernel_svm_clf.fit(X,y)
    svm_clf.append(rbf_kernel_svm_clf)

for i,svm_clf in enumerate(svm_clf):
    plt.subplot(221+i)
    plot_predictions(svm_clf,[-1.5,2.5,-1,1.5])
    plot_datasets(X,y,[-1.5,2.5,-1,1.5])
    gamma,C = hyperparams[i]
    plt.title(r'$\gamma = {}, C = {}$'.format(gamma,C),fontsize=16)

plt.figure(figsize=(10,10))
plt.show()