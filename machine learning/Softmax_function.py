# _*_ coding: utf-8 _*_
import numpy as np


# 什么是Softmax函数
"""
实际应用中，使用 Softmax 需要注意数值溢出的问题。因为有指数运算，
如果 V 数值很大，经过指数运算后的数值往往可能有溢出的可能。
所以，需要对 V 进行一些数值处理：即 V 中的每个元素减去 V 中的最大值。"""
scores = np.array([123,456,789])
scores -= np.max(scores)
p = np.exp(scores) / np.sum(np.exp(scores))
# print(p)

# Softmax损失函数
# 其中，Syi是正确类别对应的线性得分函数，Si 是正确类别对应的 Softmax输出。
# 由于 log 运算符不会影响函数的单调性，我们对 Si 进行 log 操作：
# 我们希望 Si 越大越好，即正确类别对应的相对概率越大越好，那么就可以对 Si 前面加个负号，来表示损失函数


def softmax_loss_naive(W,X,y,reg):
    """
    Softmax loss function ,naive implementation(with loops)

    Inputs have dimension D,there are C classes,and we operate on minibatches of N examples
    :param W: A numpy array of shape(D,C) containing weights
    :param X: A numpy array of shaep(N,D) containing a minibatch of data
    :param y:A numpy array of shape(N,) containing training,labels,y[i] = c means
    :param reg:(float) regularization strength
    :return: A tuple of (loss as single float, gradient with respect to weights W,an array of same shape as  W)
    """
    # initialize the loss and gradient to zero
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = X[i,:].dot(W)
        scores_shift = scores - np.max(scores)
        right_class = y[i]
        loss += (-scores_shift[right_class] + np.log(np.sum(np.exp(scores_shift))))
        for j in range(num_classes):
            softmax_output = np.exp(scores_shift[j]) / np.sum(np.exp(scores_shift))
            if j == y[i]:
                dW[:,j] += (-1 + softmax_output) * X[i,:]
            else:
                dW[:,j] += softmax_output * X[i,:]
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    dW /= num_train
    dW += reg * W

    return loss,dW

