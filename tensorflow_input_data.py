# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/17 19:17'

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
n_features = 2
n_classes = 2
batch_size = 32
h = 0.2
x,y = datasets.make_classification(n_samples=500,n_features=n_features,
                        n_redundant=0, n_informative=1,
                        n_classes=n_classes,n_clusters_per_class=1)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3)
print(train_x.shape)
print(train_y.shape)


x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
#print(xx.shape)
#print(yy.shape)
#print(np.c_[xx.ravel(), yy.ravel()])

def get_batch(x,y,batch):
    n_samples = len(x)
    for i in range(batch,n_samples,batch):
        yield x[i-batch:i], y[i-batch:i]

x_input = tf.placeholder(tf.float32,shape=[None,n_features],name='x_input')
y_input = tf.placeholder(tf.int32,shape=[None],name='y_input')

W = tf.Variable(tf.truncated_normal([n_features, n_classes]),name='W')
b = tf.Variable(tf.zeros([n_classes]),name='b')

logits = tf.sigmoid(tf.matmul(x_input,W)+b)
predict = tf.arg_max(logits,1,name='predict')
loss = tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=y_input)
loss = tf.reduce_mean(loss)
tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
acc, acc_op = tf.metrics.accuracy(labels=y_input,predictions=predict)
tf.summary.scalar('acc', acc_op)
merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('log', sess.graph)
    saver = tf.train.Saver(max_to_keep=4)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    step = 0
    for epoch in range(100): # 训练次数
        for tx,ty in get_batch(train_x,train_y, batch_size): # 得到一个batch的数据
            step += 1
            loss_value,_ ,acc_value,train_summary= sess.run([loss,optimizer,acc_op,merge_summary],feed_dict={x_input:tx,y_input:ty})
            train_writer.add_summary(train_summary,step)
            if step % 100 == 0:
                saver.save(sess, 'Model/model',global_step=step)
            print('loss = {}, acc = {}'.format(loss_value,acc_value))
    acc_value = sess.run([acc_op],feed_dict={x_input:test_x ,y_input:test_y})
    print('val acc = {}'.format(acc_value))
    prob  = sess.run([logits],feed_dict={x_input:np.c_[xx.ravel(), yy.ravel()]})
    prob  = prob [0][:,0].reshape(xx.shape)


    plt.scatter(train_x[:,0],train_x[:,1], marker='o', c=train_y,
                s=25, edgecolor='k')


    # filled contours
    cm = plt.cm.RdBu
    plt.contourf(xx, yy, prob,cmap=cm, alpha=.3)

    # contour lines
    #plt.contour(xx, yy, prob, colors='k')
    plt.show()