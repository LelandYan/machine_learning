# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/1 0:06'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
trainning_iters = 100000
batch_size = 128
display_step = 10

n_inputs = 28  # MNIST data input(image shape:28x28)
n_steps = 28  # time steps
n_hidden_unis = 128  # neurons in hidden layer
n_classes = 10  # MNIST classes (0-9 digits)

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X,weights,biases):
    # hidden layer for input to cell
    #######################################################
    # X(128 batch,28 steps,28inputs)
    X = tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,weights['in']) + biases['in']
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_unis])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)

    outputs,state = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)

    result = None
    return result
pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < trainning_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        step += 1