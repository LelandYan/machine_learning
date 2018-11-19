# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/9/17 21:17'
# import tensorflow as tf
#
# graph = tf.Graph()
# with graph.as_default():
#     foo = tf.Variable(3, name='foo')
#     bar = tf.Variable(2, name='bar')
#     result = foo + bar
#     initialize = tf.global_variables_initializer()
#
# with tf.Session(graph=graph) as sess:
#     sess.run(initialize)
#     res = sess.run(result)
# print(res)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
