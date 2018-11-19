import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(x,W)+b
y_ = tf.placeholder(tf.float32,[None,1])
cost = tf.reduce_mean(tf.pow((y_-y),2))



train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(100):
    # create fake data for actual data
    xs = np.array([[i]])
    ys = np.array([[2*i]])
    feed = {x:xs,y_:ys}
    sess.run(train_step,feed_dict=feed)
    print("After %d iteration:"%i)
    print("W: %f %", sess.run(W))
    print("b %f",sess.run(b))