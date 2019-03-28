# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/28 19:42'

import tensorflow as tf

# x = tf.Variable(3, name="x")
# y = tf.Variable(4, name="y")
# # 这句并不执行任何的计算，它只是创建一个计算图谱
# f = x * x * y + y + 2
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     result = f.eval()
# print(result)
#
# w = tf.constant(3)
# x = w + 2
# y = x + 5
# z = x * 3

# with tf.Session() as sess:
#   print(y.eval())
#   print(z.eval())

# with tf.Session() as sess:
#     y_val,z_val = sess.run([y,z])
#     print(y_val)
#     print(z_val)

# import numpy as np
# import pandas as pd
# from sklearn.datasets import fetch_california_housing
#
# # housing = pd.read_csv("housing.csv")
# housing = fetch_california_housing()
# m, n = housing.data.shape
#
# # np.c_ 按column来组合array
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
#
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="x")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)),XT), y)
# with tf.Session() as sess:
#     theta_value = theta.eval()
# print(theta_value)

# from sklearn.datasets import fetch_california_housing
# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
# housing = fetch_california_housing()
# m, n = housing.data.shape
# scaler = StandardScaler()
# scaled_housing_data = scaler.fit_transform(housing.data)
#
# # np.c_按column来组合array
# scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
# n_epochs = 1000
# learning_rate = 0.01
#
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)
# # gradients = tf.gradients(mse,[theta])[0]
# # gradients = 2/m*tf.matmul(tf.transpose(X),error)
# # training_op = tf.assign(theta,theta-learning_rate*gradients)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         if epoch % 100 == 0:
#             print("Epoch", epoch, "MSE", mse.eval())
#         sess.run(training_op)
#     best_theta = theta.eval()
import numpy as np
def my_func(a,b):
    z = 0
    for i in range(100):
        z = a * np.cos(z+i) + z * np.sin(b-i)
    return z

a = tf.Variable(0.2,name='a')
b = tf.Variable(0.3,name="b")
z = tf.Variable(0.0,name="z0")
for i in range(100):
    z = a * tf.cos(z + i) + z * tf.sin(b - i)

grads = tf.gradients(z,[a,b])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(z.eval())
    print(sess.run(grads))