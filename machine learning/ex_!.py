# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/12/22 21:49'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_table("ex1data1.txt", header=None, delimiter=",", encoding="gb2312")
m = len(data)
data_x = np.array(data)[:, 0].reshape(m, 1)
data_y = np.array(data)[:, 1].reshape(m, 1)


# the first task
# print(np.eye(5))

# the second task
plt.scatter(data_x, data_y, color='r', marker='x',label="Training Data")
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()

# the third task
def computeCost(X, y, theta):
    inner = np.power(((X * theta) - y), 2)
    return np.sum(inner / (2 * len(X)))

def gradientDescent(X,y,theta,alpha,epoch):
    cost = np.zeros(epoch)
    m = X.shape[0]
    for i in range(epoch):
        temp = theta - (alpha / m) * ((X * theta - y).T * X).T
        theta = temp
        cost[i] = computeCost(X,y,theta)
    return  theta,cost


X = np.concatenate((np.ones((m, 1)), data_x), axis=1)
y = np.matrix(data_y)
theta = np.matrix(np.zeros([2, 1]))
alpha = 0.01
epoch = 1000
final_theta,cost = gradientDescent(X,y,theta,alpha,epoch)
x = np.array(list(data_x[:,0]))
f = final_theta[0,0] + final_theta[1,0] * x
plt.scatter(data_x, data_y, color='r', marker='x',label="Training Data")
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.plot(x,f,label="Prediction")
plt.legend()
plt.show()

plt.plot(np.arange(epoch),cost,'r')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# the fourth task
