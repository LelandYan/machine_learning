# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/29 20:53'

import tensorflow as tf

a = tf.Variable(tf.constant(1,name='v'))
n = tf.constant(1)


def cond(a, n):
    return a < n


def body(a, n):
    a = a + 1
    return a, n


a, n = tf.while_loop(cond, body, [a, n])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run([a, n])
    print(res)
