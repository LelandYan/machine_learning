# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/9/20 15:20'

import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = [float(i) / 100.0 for i in range(1, 300)]
    y = [math.log(i) for i in x]
    plt.plot(x, y, 'r-', linewidth=3, label='log Curve')
    plt.plot(x[20], y[175])
    plt.plot(y[20], y[175])
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('log(x)')
    plt.show()
