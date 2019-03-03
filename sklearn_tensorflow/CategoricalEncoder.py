# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/3 12:36'

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse


import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.r_[a,b])
print(np.c_[a,b])