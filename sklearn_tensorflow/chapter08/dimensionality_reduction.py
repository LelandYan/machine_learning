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
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)


# PCA using SVD decomposition
X_centered = X - X.mean(axis=0)
U,s,Vt = np.linalg.svd(X_centered)
# c1 = Vt.T[:,0]
# c2 = Vt.T

m,n = X.shape
S = np.zeros(X_centered.shape)
S[:n,:n] = np.diag(s)
# 判断是否X_centered,U.dot(S).dot(Vt)相似
np.allclose(X_centered,U.dot(S).dot(Vt))
W2 = Vt.T[:,:2]
X2D_using_svd = X_centered.dot(W2)
# print(X_centered.shape)
# print(X2D.shape)

# PCA using Scikit-Learn
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print(np.allclose(X2D,X2D_using_svd))


X3D_inv = pca.inverse_transform(X2D)
print(np.allclose(X3D_inv,X))

# 即使使用pca的inverse_transform也不能返回原来的全部信息
