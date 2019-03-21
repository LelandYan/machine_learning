# _*_ coding: utf-8 _*_
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置全局的随机种子
np.random.seed(42)
mpl.rc("axes",labelsize=14)
mpl.rc("xtick",labelsize=12)
mpl.rc("ytick",labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"

def save_fig(fig_id,tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR,"images",CHAPTER_ID,".png")
    print("Saving figure",fig_id)
    if tight_layout:
        # tight_layout会自动调整子图参数，使之填充整个图像区域
        plt.tight_layout()
    plt.savefig(path,format="png",dpi=300)

import warnings
warnings.filterwarnings(action="ignore",message="internal gelsd")

np.random.seed(42)
# m = 60
# w1, w2 = 0.1, 0.3
# noise = 0.1
#
# angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
# X = np.empty((m, 3))
# X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
# X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
# X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
#
#
# # PCA using SVD decomposition
# X_centered = X - X.mean(axis=0)
# U,s,Vt = np.linalg.svd(X_centered)
# # c1 = Vt.T[:,0]
# # c2 = Vt.T
#
# m,n = X.shape
# S = np.zeros(X_centered.shape)
# S[:n,:n] = np.diag(s)
# # 判断是否X_centered,U.dot(S).dot(Vt)相似
# np.allclose(X_centered,U.dot(S).dot(Vt))
# W2 = Vt.T[:,:2]
# X2D_using_svd = X_centered.dot(W2)
# # print(X_centered.shape)
# # print(X2D.shape)
#
# # PCA using Scikit-Learn
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
# X2D = pca.fit_transform(X)
# print(np.allclose(X2D,X2D_using_svd))
#
#
# X3D_inv = pca.inverse_transform(X2D)
# print(np.allclose(X3D_inv,X))
#
# # 即使使用pca的inverse_transform也不能返回原来的全部信息
# print("the reconstruction error",np.mean(np.sum(np.square(X3D_inv - X), axis=1)))
#
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d
#
# class Arrow3D(FancyArrowPatch):
#     def __init__(self,xs,ys,zs,*args,**kwargs):
#         FancyArrowPatch.__init__(self,(0,0),(0,0),*args,**kwargs)
#         self._verts3d = xs,ys,zs
#
#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         FancyArrowPatch.draw(self, renderer)
#
# axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]
#
# x1s = np.linspace(axes[0], axes[1], 10)
# x2s = np.linspace(axes[2], axes[3], 10)
# x1, x2 = np.meshgrid(x1s, x2s)
#
# C = pca.components_
# R = C.T.dot(C)
# z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])
#
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(figsize=(6, 3.8))
# ax = fig.add_subplot(111, projection='3d')
#
# X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
# X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]
#
# ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)
#
# ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
# np.linalg.norm(C, axis=0)
# ax.add_artist(Arrow3D([0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
# ax.add_artist(Arrow3D([0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
# ax.plot([0], [0], [0], "k.")
#
# for i in range(m):
#     if X[i, 2] > X3D_inv[i, 2]:
#         ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
#     else:
#         ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#505050")
#
# ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
# ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
# ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
# ax.set_xlabel("$x_1$", fontsize=18)
# ax.set_ylabel("$x_2$", fontsize=18)
# ax.set_zlabel("$x_3$", fontsize=18)
# ax.set_xlim(axes[0:2])
# ax.set_ylim(axes[2:4])
# ax.set_zlim(axes[4:6])
#
# # Note: If you are using Matplotlib 3.0.0, it has a bug and does not
# # display 3D graphs properly.
# # See https://github.com/matplotlib/matplotlib/issues/12239
# # You should upgrade to a later version. If you cannot, then you can
# # use the following workaround before displaying each 3D graph:
# # for spine in ax.spines.values():
# #     spine.set_visible(False)
#
# save_fig("dataset_3d_plot")
# plt.show()
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal')
#
# ax.plot(X2D[:, 0], X2D[:, 1], "k+")
# ax.plot(X2D[:, 0], X2D[:, 1], "k.")
# ax.plot([0], [0], "ko")
# ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
# ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
# ax.set_xlabel("$z_1$", fontsize=18)
# ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
# ax.axis([-1.5, 1.3, -1.2, 1.2])
# ax.grid(True)
# save_fig("dataset_2d_plot")
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
#
# X,t = make_swiss_roll(n_samples=1000,noise=0.2,random_state=42)
# axes = [-11.5, 14, -2, 23, -12, 15]
#
# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
# ax.view_init(10, -70)
# ax.set_xlabel("$x_1$", fontsize=18)
# ax.set_ylabel("$x_2$", fontsize=18)
# ax.set_zlabel("$x_3$", fontsize=18)
# ax.set_xlim(axes[0:2])
# ax.set_ylim(axes[2:4])
# ax.set_zlim(axes[4:6])
#
# save_fig("swiss_roll_plot")
# # plt.show()
# plt.figure(figsize=(11,4))
# plt.subplot(121)
# plt.scatter(X[:,0],X[:,1],c=t,cmap=plt.cm.hot)
# plt.axis(axes[:4])
# plt.xlabel("$x_1$",fontsize=18)
# plt.ylabel("$x_2$",fontsize=18.,rotation=0)
# plt.grid(True)
#
# plt.subplot(122)
# plt.scatter(t,X[:,1],c=t,cmap=plt.cm.hot)
# plt.axis([4, 15, axes[2], axes[3]])
# plt.xlabel("$z_1$", fontsize=18)
# plt.grid(True)
#
# save_fig("squished_swiss_roll_plot")
# plt.show()

from scipy.io import loadmat

mnist = loadmat("mnist-original.mat")
X = mnist["data"].T
y = mnist["label"][0]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.decomposition import PCA
# pca = PCA()
# pca.fit(X_train)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
# print("降维的维数")

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
# print(pca.n_components_)
X_recovered = pca.inverse_transform(X_reduced)


def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("Compressed", fontsize=16)
plt.show()
# save_fig("mnist_compression_plot")

# 增量PCA（Incrementtal PCA）
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print(".", end="") # not shown in the book
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced)
plt.figure(figsize=(7,4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.subplot(122)
plot_digits(X_recovered_inc_pca[::2100])
plt.tight_layout()
plt.show()

# Using memmap()
filename = 'mnist-original.mat'
m,n = X_train.shape
X_mm = np.memmap(filename,dtype="float32",mode="write",shape=(m,n))

