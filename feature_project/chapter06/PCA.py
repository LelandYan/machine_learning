# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/28 15:42'

from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load the data
digits_data = datasets.load_digits()
n = len(digits_data.images)

# Each image is represented as an 8-by-8 array
# Flatten this array as input to PCA
image_data = digits_data.images.reshape((n, -1))

# image_data.shape
labels = digits_data.target
pca_transformer = PCA(n_components=0.8)
# at least 80% of the total variance
pca_images = pca_transformer.fit_transform(image_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(100):
    ax.scatter(pca_images[i, 0], pca_images[i, 1], pca_images[i, 2])
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
ax.set_zlabel('Principal component 3')
plt.show()