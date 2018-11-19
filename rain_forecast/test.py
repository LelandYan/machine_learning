import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

CSV_FILE_PATH = '01.csv'
df = pd.read_csv(CSV_FILE_PATH)
shapes = df.values.shape
data = df.values[:, 0:shapes[1] - 3]
result = df.values[:, shapes[1] - 3:shapes[1] - 2]
train_x, test_x, train_y, test_y = train_test_split(data, result, test_size=0.3)
n_features = train_x.shape[1]
train_y = np.array(train_y.flatten())
test_y = np.array(test_y.flatten())


def get_batch(x, y, batch):
    n_samples = len(x)
    for i in range(batch, n_samples, batch):
        yield x[i - batch:i], y[i - batch:i]


n_classes = 1
batch_size = 200
x_input = tf.placeholder(tf.float32, shape=[None, n_features], name='x_input')
y_input = tf.placeholder(tf.int32, shape=[None], name='y_input')

W = tf.Variable(tf.truncated_normal([n_features, n_classes]), name='W')
b = tf.Variable(tf.zeros([10]) + 0.1, name='b')

logits = tf.sigmoid(tf.matmul(x_input, W) + b)


predict = tf.arg_max(logits, 1, name='predict')
loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=y_input)
loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
acc, acc_op = tf.metrics.accuracy(labels=y_input, predictions=predict)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    step = 0
    for epoch in range(10):  # 训练次数
        for tx, ty in get_batch(train_x, train_y, batch_size):  # 得到一个batch的数据
            step += 1
            loss_value, _, acc_value = sess.run([loss, optimizer, acc_op], feed_dict={x_input: tx, y_input: ty})
            # print('loss = {}, acc = {}'.format(loss_value, acc_value))
    acc_value = sess.run([acc_op], feed_dict={x_input: test_x, y_input: test_y})
    print('val acc = {}'.format(acc_value))