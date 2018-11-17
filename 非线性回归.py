import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def gamma(x):
    q = 1
    while True:
        if x - 1 >= 1:
            x = x - 1
            q = q * x
        else:
            break
    return q


def pearson(j, k, l, x):# j - a k - b a0 -l
    j = 1
    k = 1
    l = 1
    a = [j, k, l]
    s = a[1] ** a[2]
    s = s / gamma(a[0])
    s = s * (x - a[2]) ** (a[0] - 1)
    s = s * math.exp(-a[1] * (x - a[2]))
    return s
# 使用numpy生成200个随机点
x_data = np.linspace(-1, 1, 200)[:, np.newaxis]

noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.power(x_data,4)+np.sin(x_data)

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 构建神经网络中间层
Weights_l1 = tf.Variable(tf.random_normal([1, 10]))  # 1 10
biases_l1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_l1 = tf.matmul(x, Weights_l1) + biases_l1
Li = tf.nn.tanh(Wx_plus_b_l1)

# 定义神经网络输出层
Weights_l2 = tf.Variable(tf.random_normal([10, 20]))
biases_l2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_l2 = tf.matmul(Li, Weights_l2) + biases_l2
prediction1 = tf.nn.tanh(Wx_plus_b_l2)

Weights_l3 = tf.Variable(tf.random_normal([20, 10]))
biases_l3 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_l3 = tf.matmul(prediction1, Weights_l3) + biases_l3
prediction = tf.nn.tanh(Wx_plus_b_l3)

# 二次代价函数(方差)
loss = tf.reduce_mean(tf.square(y - prediction))
# 梯度下降法
train_stop = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量的初始化
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(train_stop, feed_dict={x: x_data, y: y_data})
    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
