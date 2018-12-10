from scipy.special import gamma
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


def gamma(x):
    q = 1
    while True:
        if x - 1 >= 1:
            x = x - 1
            q = q * x
        else:
            break
    return q


# 定义伽马函数
def pearson(j, k, l, x):
    a = [j, k, l]
    # j k l 分别表示α β a0三个参数
    s = a[1] ** a[0]
    s = s / gamma(a[0])
    y = (x - a[2]) ** (a[0] - 1)
    s = s * y
    b = -a[1] * (x - a[2])
    s = s * math.exp(b)
    return s


# 完整定义皮尔逊Ⅲ型曲线
a = []
for j in np.arange(1, 2, .1):
    for k in np.arange(2, 3, .1):
        for l in np.arange(.1, 1, .1):
            for i in np.arange(1, 21, 1):
                a.append(pearson(j, k, l, i))
b = []
for x in range(900):
    b.append(a[20 * x:20 + 20 * x])
# b用来装纵坐标对应的值，列表套列表
c = {}
for i in range(900):
    c[i] = b[i]
# c为b中的每个列表加上标签，c是一个字典
d = []
for j in np.arange(1, 2, .1):
    for k in np.arange(2, 3, .1):
        for l in np.arange(.1, 1, .1):
            d.append([j, k, l])
# d用来装α β a0三个参数，列表套列表
e = {}
for i in range(900):
    e[i] = d[i]
# e为d中的每个列表加上标签，e也是一个字典
u = np.array(b)[:850, :]
i = np.array(d)[:850, :]
o = np.array(b)[850:, :]
p = np.array(d)[850:, :]

x = tf.placeholder(tf.float32, [1, 20])
y = tf.placeholder(tf.float32, [1, 3])
# z=tf.placeholder(tf.float32,[1,20])

WeightL = tf.Variable(tf.random_normal([20, 1]))
BasisL = tf.Variable(tf.zeros([1, 1]))
L = tf.matmul(x, WeightL) + BasisL
taL = tf.nn.tanh(L)

# z中盛放预测结果
WeightL1 = tf.Variable(tf.random_normal([1, 4]))
BasisL1 = tf.Variable(tf.zeros([1, 4]))
L1 = tf.matmul(taL, WeightL1) + BasisL1
taL1 = tf.nn.tanh(L1)

WeightL2 = tf.Variable(tf.random_normal([4, 3]))
BasisL2 = tf.Variable(tf.zeros([1, 3]))
L2 = tf.matmul(taL1, WeightL2) + BasisL2
taL2 = tf.nn.tanh(L2)
# 简单神经网络构建

init = tf.global_variables_initializer()
# 初始化参数
opti = tf.train.GradientDescentOptimizer(0.3).minimize(tf.reduce_mean(tf.square(y - taL2)))
with tf.Session() as sess:
    sess.run(init)
    x_in = sess.run(tf.convert_to_tensor(u))
    y_in = sess.run(tf.convert_to_tensor(i))
    x_intt = sess.run(tf.convert_to_tensor(o))
    y_intt = sess.run(tf.convert_to_tensor(p))
    for i in range(850):
        sess.run(opti, feed_dict={x: (x_in[i]).reshape(1,20), y: (y_in[i]).reshape(1,3)})
    for i in range(850, 900):
        pass
        # data = sess.run(result, feed_dict={x: x_intt[i], z: z_intt[i]})
