# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/18 18:31'

from sklearn.datasets import load_iris
import matplotlib as mpl
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

# plt.subplot(121)
# plt.plot(X[y == 0, 2], X[y == 0, 3], "yo", label="Iris-Setosa")
# plt.plot(X[y == 1, 2], X[y == 1, 3], "bs", label="Iris-Versicolor")
# plt.plot(X[y == 2, 2], X[y == 2, 3], "g^", label="Iris-Virginica")
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(fontsize=12)
#
# plt.subplot(122)
# plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
# plt.xlabel("Petal length", fontsize=14)
# plt.tick_params(labelleft=False)
#
# plt.show()

from sklearn.mixture import GaussianMixture
import numpy as np

# y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
# mapping = np.array([2, 0, 1])
# y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])
# plt.plot(X[y_pred == 0, 2], X[y_pred == 0, 3], "yo", label="Cluster 1")
# plt.plot(X[y_pred == 1, 2], X[y_pred == 1, 3], "bs", label="Cluster 2")
# plt.plot(X[y_pred == 2, 2], X[y_pred == 2, 3], "g^", label="Cluster 3")
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(loc="upper left", fontsize=12)
# # plt.show()
# print(np.sum(y_pred == y) / len(y_pred))

# K-Means
from sklearn.datasets import make_blobs

blob_centers = np.array([
    [0.2, 2.3],
    [-1.5, 2.3],
    [-2.8, 1.8],
    [-2.8, 2.8],
    [-2.8, 1.3]
])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X,y = make_blobs(n_samples=2000,centers=blob_centers,cluster_std=blob_std,random_state=7)
def plot_cluster(X,y=None):
    plt.scatter(X[:,0],X[:,1],c=y,s=1)
    plt.xlabel('$x_1$',fontsize=14)
    plt.ylabel('$x_2$',fontsize=14,rotation=0)

# plt.figure(figsize=(8,4))
# plot_cluster(X)
# plt.show()

from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k,random_state=42)
y_pred = kmeans.fit_predict(X)
print(kmeans.cluster_centers_)
print(y_pred == kmeans.labels_)

# Decision Boundaries
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundary(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

# plt.figure(figsize=(8, 4))
# plot_decision_boundary(kmeans, X)
# plt.show()
#
# kmeans_iter1 = KMeans(n_clusters=5,init='random',n_init=1,algorithm='full',max_iter=1,random_state=1)
# kmeans_iter2 = KMeans(n_clusters=5,init='random',n_init=1,algorithm='full',max_iter=2,random_state=1)
# kmeans_iter3 = KMeans(n_clusters=5,init='random',n_init=1,algorithm='full',max_iter=3,random_state=1)
#
# kmeans_iter1.fit(X)
# kmeans_iter2.fit(X)
# kmeans_iter3.fit(X)
# plt.figure(figsize=(10, 8))
#
# plt.subplot(321)
# plot_data(X)
# plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
# plt.ylabel("$x_2$", fontsize=14, rotation=0)
# plt.tick_params(labelbottom=False)
# plt.title("Update the centroids (initially randomly)", fontsize=14)
#
# plt.subplot(322)
# plot_decision_boundary(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
# plt.title("Label the instances", fontsize=14)
#
# plt.subplot(323)
# plot_decision_boundary(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
# plot_centroids(kmeans_iter2.cluster_centers_)
#
# plt.subplot(324)
# plot_decision_boundary(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)
#
# plt.subplot(325)
# plot_decision_boundary(kmeans_iter2, X, show_centroids=False)
# plot_centroids(kmeans_iter3.cluster_centers_)
#
# plt.subplot(326)
# plot_decision_boundary(kmeans_iter3, X, show_ylabels=False)

def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundary(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundary(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)

# kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1,
#                          algorithm="full", random_state=11)
# kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1,
#                          algorithm="full", random_state=19)
#
# plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,
#                           "Solution 1", "Solution 2 (with a different random init)")
#
# plt.show()

from sklearn.cluster import MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=5,random_state=42)
minibatch_kmeans.fit(X)
