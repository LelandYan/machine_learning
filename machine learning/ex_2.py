# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/12/24 22:56'

import numpy as np
import matplotlib.pyplot as plt

fileName = "ex2data1.txt"
data = np.loadtxt(fileName, delimiter=",")
m = data.shape[0]  # 读取样本的数量
bool_type = np.ravel(data.T[-1:])

pos = np.array([data[i] for i in range(m) if bool_type[i] == 1])
neg = np.array([data[i] for i in range(m) if bool_type[i] == 0])

plt.figure()
plt.plot(pos[:, 0], pos[:, 1], 'k+', label='Admitted')
plt.plot(neg[:, 0], neg[:, 1], 'yo', label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()
