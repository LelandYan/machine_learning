# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/17 21:38'

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


class Neural_Network(object):
    def get_batch(self, x, y, batch):
        n_samples = len(x)
        for i in range(batch, n_samples, batch):
            yield x[i - batch:i], y[i - batch:i]

    def __int__(self, data, result):
        self.n_classes = 2
        self.batch_size = 10
        self.data = data
        self.result = result
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data, self.result, test_size=0.3)
        self.n_features = self.train_x.shape[1]
        self.train_y = np.array(self.train_y.flatten())
        self.test_y = np.array(self.test_y.flatten())

        self.x_input = tf.placeholder(tf.float32, shape=[None, self.n_features], name='x_input')
        self.y_input = tf.placeholder(tf.int32, shape=[None], name='y_input')

        W1 = tf.Variable(tf.truncated_normal([self.n_features, 10]), name='W1')
        b1 = tf.Variable(tf.zeros([10]) + 0.1, name='b1')

        logits1 = tf.sigmoid(tf.matmul(self.x_input, W1) + b1)

        W = tf.Variable(tf.truncated_normal([10, self.n_classes]), name='W2')
        b = tf.Variable(tf.zeros([self.n_classes]), name='b2')

        logits = tf.nn.softmax(tf.matmul(logits1, W) + b)  # 优化一
        predict = tf.arg_max(logits, 1, name='predict')
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.y_input)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self.y_input, predictions=predict)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            step = 0
            for epoch in range(200):  # 训练次数
                for self.tx, self.ty in self.get_batch(self.train_x, self.train_y, self.batch_size):  # 得到一个batch的数据
                    step += 1
                    loss_value, _, acc_value = sess.run([self.loss, self.optimizer, self.acc_op],
                                                        feed_dict={self.x_input: self.tx, self.y_input: self.ty})
                    # print('loss = {}, acc = {}'.format(loss_value, acc_value))
            acc_value = sess.run([self.acc_op], feed_dict={self.x_input: self.test_x, self.y_input: self.test_y})
            # print('val acc = {}'.format(acc_value))
            return (acc_value)
