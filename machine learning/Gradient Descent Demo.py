# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/10/10 23:20'
import numpy as np
import matplotlib.pyplot as plt  # plt

# 李宏毅原代码没有加载相关参数，以上为我自行加载的。

x_data = [338, 333, 328, 207, 226, 25, 179, 60, 208, 606]
y_data = [640, 633, 619, 393, 428, 27, 193, 66, 226, 1591]

x = np.arange(-200, -100, 1)  # bias
y = np.arange(-5, 5, 0.1)  # weight
Z = np.zeros((len(x), len(y)))  # loss
X, Y = np.meshgrid(x, y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n]) ** 2
        Z[j][i] = Z[j][i] / len(x_data)

# y_data = b + w*x_data
b = -120  # initial b
w = -4  # initial w
lr = 0.0001  # learning rate
iteration = 100000

# store initial values for plotting
b_history = [b]
w_history = [w]

lr_w = 0
lr_b = 0
# iterations
for i in range(iteration):

    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        # 这里是对loss函数（平方差）求导得到最好的我w，b
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]
    lr_w = lr_w + w_grad ** 2
    lr_b = lr_b + b_grad ** 2
    # update parameters
    b = b - lr / np.sqrt(lr_b) * b_grad
    w = w - lr / np.sqrt(lr_w) * w_grad

    # store parameters for plotting
    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
# expected goal
plt.plot([-188.4], [2.67], 'x', ms=6, marker=6, color='orange')
# ms和marker分别代表指定点的长度和宽度。
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
