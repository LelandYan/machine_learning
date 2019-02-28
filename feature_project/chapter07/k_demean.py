# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/28 16:21'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# n_data = 1000
# seed = 1
# n_centers = 4
#
# # 产生高斯随机数，运行k-demean
# blobs, blob_labels = make_blobs(n_samples=n_data, n_features=2, centers=n_centers, random_state=seed)
# clusters_blob = KMeans(n_clusters=n_centers, random_state=seed).fit_predict(blobs)
#
# # 产生随机数，运行K-demean
# uniform = np.random.rand(n_data, 2)
# clusters_uniform = KMeans(n_clusters=n_centers, random_state=seed).fit_predict(uniform)
#
# # 可视化
# figure = plt.figure()
# plt.subplot(221)
# plt.scatter(blobs[:, 0], blobs[:, 1], c=blob_labels, cmap='gist_rainbow')
# plt.title("(a) Four randomly generated blobs", fontsize=14)
# plt.axis('off')
#
# plt.subplot(222)
# plt.scatter(blobs[:, 0], blobs[:, 1], c=clusters_blob, cmap='gist_rainbow')
# plt.title("(b) Clusters found via K-means", fontsize=14)
# plt.axis('off')
#
# plt.subplot(223)
# plt.scatter(uniform[:, 0], uniform[:, 1])
# plt.title("(c) 1000 randomly generated points", fontsize=14)
# plt.axis('off')
#
# plt.subplot(224)
# plt.scatter(uniform[:, 0], uniform[:, 1], c=clusters_uniform, cmap='gist_rainbow')
# plt.title("(d) Clusters found via K-means", fontsize=14)
# plt.axis('off')
# plt.show()

# 曲面拼接聚类
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold,datasets

# 在瑞士卷训练集上产生噪声
X,color = datasets.samples_generator.make_swiss_roll(n_samples=1500)

# 用100 K-均值聚类来估计数据集
clusters_swiss_roll = KMeans(n_clusters=100,random_state=1).fit_predict(X)

# 展示用数据集，其中颜色是K-均值聚类的id
fig2 = plt.figure()
ax = fig2.add_subplot(111,projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=clusters_swiss_roll, cmap='Spectral')
plt.show()