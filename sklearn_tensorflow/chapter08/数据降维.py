# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/21 15:44'
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# to make this notebook output stable across runs
np.random.seed(42)

mpl.rc("axes",labelsize=14)
mpl.rc("xtick",labelsize=12)
mpl.rc("ytick",labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"

def save_fig(fig_id,tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR,"images",fig_id,".png")
    print("Saving figure",fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format="png",dpi=300)

import warnings
warnings.filterwarnings(action="ignore",message="internal gelsd")


np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)


# X_centered = X - X.mean(axis=0)
# U,s,Vt = np.linalg.svd(X_centered)
# print("X_centered",X_centered.shape)
# # print("U",U.shape)
# # print("s",s.shape)
# # print("Vt",Vt.shape)
# # print(s)
# c1 = Vt.T[:,0]
# c2 = Vt.T[:,1]
# m,n = X.shape
# S = np.zeros((X_centered.shape))
# S[:n,:n] = np.diag(s)
# res = np.allclose(X_centered,U.dot(S).dot(Vt))
#
# W2  = Vt.T[:,:2]
# X2D = X_centered.dot(W2)
# X2D_using_svd = X2D
#
# # PCA using Scikit-Learn
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X2D = pca.fit_transform(X)
# # print(X2D[:5])
# X3D_inv = pca.inverse_transform(X2D)
# np.allclose(X3D_inv,X)
#
# print(pca.components_)
# print(pca.explained_variance_ratio_)

# Mainifold Learning
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

X,t = make_swiss_roll(n_samples=1000,noise=0.2,random_state=42)
axes = [-11.5,14,-2,23,-12,15]
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111,projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

plt.show()