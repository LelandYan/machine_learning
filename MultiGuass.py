# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/9/30 20:13'

# import time
# from scipy.optimize import leastsq
# import scipy.optimize as opt
# import scipy
# import matplotlib.pyplot as plt
# from scipy.stats import norm,poisson

import numpy as np
from scipy import stats
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def normal_distribution():
    mu = 0
    sigma = 1
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50)
    y = np.exp(-(x - mu) ** 2 / 2 * sigma ** 2) / (np.sqrt(2 * np.pi) * sigma)
    print(x.shape)
    print(y.shape)
    plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8)
    plt.grid(True)
    plt.title('Guass分布')
    plt.show()


if __name__ == '__main__':
    a = np.arange(0, 60, 10)  # 最后10表示表示步长
    d = np.logspace(1, 2, 10, endpoint=True)  # 等比数列
    c = np.linspace(1, 10, 10, endpoint=False)  # 最后10表示10个元素
    print(d)
    print(c)
    print(a)
    l = [1, 2, 3, 4, 5, 6]
    b = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], dtype=np.float)
    print(b.shape)
    print(b)
    b.shape = 6, 3  # 修改原始变量
    print(b)
    c = b.reshape((2, 9))  # 修改变量
    print(c)
    a = np.random.rand(10)
    print(a)
    print(a > 0.5)
    print(a[a > 0.5])
    c = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
    print(c)
    print(c[(0, 1, 2, 3), (2, 3, 4, 5)])
    print(c[3:, [0, 2, 5]])
    normal_distribution()
