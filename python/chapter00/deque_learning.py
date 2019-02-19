# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/15 12:33'
from collections import deque

from queue import Queue

from collections import Counter

a = ['bobby1', 'bobby2', 'bobby2', 'bobby1''bobby1']
# print(Counter(a).most_common(2))

from collections import OrderedDict
# user_dict = OrderedDict()
# user_dict.move_to_end("key")
# user_dict.popitem() 返回key与value
# user_dict.pop() 返回value，而且要传入key

from collections import ChainMap
from urllib.parse import parse_qs

generator = (x for x in [1, 2, 3])
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
# print(iris.data.shape)
# print(iris.target)
X = iris.data[:, :2]
import matplotlib.pyplot as plt
# plt.scatter(X[:,0],X[:,1])
# plt.show()
# y = iris.target
# plt.scatter(X[y==0,0],X[y==0,1],color='red',marker='o')
# plt.scatter(X[y==1,0],X[y==1,1],color='blue',marker='+')
# plt.scatter(X[y==2,0],X[y==2,1],color='green',marker='x')
# plt.show()
# from collections import Counter
#
# # 返回最大的一个元素
# c = Counter.most_common(1)
from datetime import datetime


def log(message, when=None):
    when = datetime.now() if when is None else when
    print("{}: {}".format(when, message))


import time


def safe_division(number, divisor, ignore_overflow, ignore_zero_division):
    try:
        return number / divisor
    except OverflowError:
        if ignore_overflow:
            return 0
        else:
            raise
    except ZeroDivisionError:
        if ignore_zero_division:
            return float("inf")
        else:
            raise


if __name__ == '__main__':
    # log("Hi there")
    # time.sleep(0.1)
    # log("Hi again")
    result = safe_division(1,0,False,True)
    print(result)
