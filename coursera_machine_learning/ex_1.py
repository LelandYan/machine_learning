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
# plt.scatter(data_x, data_y, color='r', marker='x',label="Training Data")
# plt.ylabel('Profit in $10,000s')
# plt.xlabel('Population of City in 10,000s')
# plt.show()

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

def normal_equations(X,y):
    theta = np.linalg.inv(X.T*X)*X.T*y
    return theta

X = np.matrix(np.concatenate((np.ones((m, 1)), data_x), axis=1))
y = np.matrix(data_y)
theta = np.matrix(np.zeros([2, 1]))
alpha = 0.01
epoch = 1000
final_theta,cost = gradientDescent(X,y,theta,alpha,epoch)
print(final_theta)
final_theta = normal_equations(X,y)
print(final_theta)
x = np.array(list(data_x[:,0]))
f = final_theta[0,0] + final_theta[1,0] * x

# plt.scatter(data_x, data_y, color='r', marker='x',label="Training Data")
# plt.ylabel('Profit in $10,000s')
# plt.xlabel('Population of City in 10,000s')
# plt.plot(x,f,label="Prediction")
# plt.legend()
# plt.show()


# plt.plot(np.arange(epoch),cost,'r')
# plt.xlabel("Iteration")
# plt.ylabel("Cost")
# plt.show()

theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.matrix(np.array([theta0_vals[i],theta1_vals[j]]).reshape((2,1)))
        J_vals[i,j] = computeCost(X,y,t)


######################################## 3D图 #######################################
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(theta0_vals,theta1_vals,J_vals,rstride=1,cstride=1,cmap="rainbow")
# plt.xlabel('theta_0')
# plt.ylabel('theta_1')
# plt.show()
######################################## 3D图 #######################################

######################################## 等高线图 #######################################
# plt.figure()
# plt.contourf(theta0_vals, theta1_vals, J_vals, 20, alpha = 0.6, cmap = plt.cm.hot)
# a = plt.contour(theta0_vals, theta1_vals, J_vals, colors = 'black')
# plt.clabel(a,inline=1,fontsize=10)
# plt.plot(theta[0,0],theta[1,0],'r',marker='x')
# plt.show()
######################################## 等高线图 #######################################

data2 = pd.read_table("ex1data2.txt", header=None, delimiter=",", encoding="gb2312")

data2 = (data2 - data2.mean()) / data2.std()
m = len(data2)
data_x = np.array(data2)[:, 0:2].reshape(m, 2)
data_y = np.array(data2)[:, 2].reshape(m, 1)
data_x = np.concatenate((np.zeros((m,1)),data_x),axis=1)
X = np.matrix(data_x)
y = np.matrix(data_y)
theta = np.matrix(np.zeros((3,1)))
final_theta,cost = gradientDescent(X,y,theta,alpha,epoch)

plt.plot(np.arange(epoch),cost,'r')
plt.show()