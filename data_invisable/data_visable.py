# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/1 20:19'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 将文件名改为对应的csv文件名，运行程序
data = pd.read_csv("01.csv")
fpa = data.iloc[:,1][1:201]
fbe = data.iloc[:,6][1:201]
fbe = np.array(fbe,dtype=np.float64)
cfpa = data.iloc[:,8][1:201]
cfpa = np.array(cfpa,dtype=np.float64)
ran = np.arange(1,201)
plt.plot(ran,np.array(fpa))
plt.plot(ran,np.array(fbe))
plt.plot(ran,np.array(cfpa))
plt.show()