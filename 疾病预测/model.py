# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/17 21:38'

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
class Neural_Network(object):
    def __int__(self):
        self.n_classes = 2
        self.batch_size = 10
        self.df = pd.read_csv('csv_result-ALL-AML_train.csv')
        self.shapes = self.df.values.shape
        self.data = self.df.values[:, 1:self.shapes[1] - 1]
        self.result = self.df.values[:, self.shapes[1] - 1:self.shapes[1]]
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data, self.result, test_size=0.3)
        self.n_features = self.train_x.shape[1]
        self.train_y = np.array(self.train_y.flatten())
        self.test_y = np.array(self.test_y.flatten())

    def get_batch(self,x, y, batch):
        n_samples = len(x)
        for i in range(batch, n_samples, batch):
            yield x[i - batch:i], y[i - batch:i]

    def variable(self):
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.n_features], name='x_input')
        self.y_input = tf.placeholder(tf.int32, shape=[None], name='y_input')

        W1 = tf.Variable(tf.truncated_normal([self.n_features, 10]), name='W')
        b1 = tf.Variable(tf.zeros([10]) + 0.1, name='b')

        logits1 = tf.sigmoid(tf.matmul(self.x_input,W1) + b1)

        W = tf.Variable(tf.truncated_normal([10, self.n_classes]), name='W')
        b = tf.Variable(tf.zeros([self.n_classes]), name='b')

        logits = tf.sigmoid(tf.matmul(logits1,W) + b)
        predict = tf.arg_max(logits, 1, name='predict')
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.y_input)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self.y_input, predictions=predict)
    def model(self):
        time.sleep(10)
        self.variable()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            step = 0
            for epoch in range(200):  # 训练次数
                for self.tx, self.ty in self.get_batch(self.train_x, self.train_y, self.batch_size):  # 得到一个batch的数据
                    step += 1
                    loss_value, _, acc_value = sess.run([self.loss, self.optimizer, self.acc_op], feed_dict={self.x_input: self.tx, y_input: self.ty})
                    print('loss = {}, acc = {}'.format(loss_value, acc_value))
            acc_value = sess.run([self.acc_op], feed_dict={self.x_input: self.test_x, self.y_input: self.test_y})
            print('val acc = {}'.format(acc_value))

if __name__ == '__main__':
    model = Neural_Network()
    model.model()