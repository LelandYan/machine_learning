import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 100  ###
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义俩个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.01, dtype=tf.float32)
# 创建一个简单的神经网路
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))  ###
b1 = tf.Variable(tf.zeros([500]) + 0.1)  ###
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))  ###
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))  ###
b3 = tf.Variable(tf.zeros([10]) + 0.1)  ###

prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 二次代价函数 （直线函数）
# loss = tf.reduce_mean(tf.square(y - prediction))
# 交叉熵函数（s型函数）
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用梯度下降法
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
# 结果存放在一个布尔值列表行，返回的时最大值返回的位置
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        # train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        learning_rate = sess.run(lr)
        print("Iter" + str(epoch) + "Testing Accuracy " + str(test_acc) + "learning_rate " + str(learning_rate))
