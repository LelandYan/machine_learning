# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/5 23:21'
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 导入数据
mnist = loadmat("mnist-original.mat")
X = mnist["data"].T
y = mnist["label"][0]

# some_digit = X[0]
# some_digit_image = some_digit.reshape(28,28)
# plt.imshow(some_digit_image,cmap=mpl.cm.binary,interpolation="nearest")
# plt.axis('off')
# plt.show()
# print(y[0])
y = y.astype(np.uint8)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary)
    plt.axis('off')


# EXTRA
def plot_digits(instances, image_per_row=10, **options):
    size = 28
    # 一行显示有多少张图片
    image_per_row = min(len(instances), image_per_row)
    # 将图片变为28*28形状
    images = [instance.reshape(size, size) for instance in instances]

    n_rows = (len(instances) - 1) // image_per_row + 1
    row_images = []
    # 剩余还有几个空余的位置
    n_empty = n_rows * image_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * image_per_row:(row + 1) * image_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


plt.figure(figsize=(9, 9))
example_images = X[:100]
plot_digits(example_images, image_per_row=10)
plt.show()
