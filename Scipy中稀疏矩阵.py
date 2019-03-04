# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/24 22:24'

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
# x = np.linspace(0,1e6,10)
# plt.plot(x,8.0*(x**2)/1e6,lw=5)
# plt.xlabel('size n')
# plt.ylabel('memory [MB]')

# x = np.array([[0,0,1],[2,0,1],[0,3,0]])
# mtx = sparse.lil_matrix(x)
# user_item_pairs = np.asarray(mtx.nonzero())
# print(mtx.todense())
# print(user_item_pairs)
# line = np.arange(5)
# print(line)
# np.random.shuffle(line)
# print(line)

# plt.spy(x)
# plt.show()
# mtx = sparse.dok_matrix((5,5))
# print(mtx.todense())
# user_item_matrix = np.loadtxt("DrDiAssMat.txt")
# n_users, n_items = user_item_matrix.shape
# # print(n_items,n_users)
# mtx = sparse.lil_matrix(user_item_matrix)
# print(mtx.todense())
# print(mtx.rows[1])
# print(mtx.rows[0])
# from tqdm import tqdm
#
# for i in tqdm(range(1000000)):
#     print(i)

# x = np.array([[1,2,3],
#               [4,5,6]])
# np.random.shuffle(x)
# print(x)
# matrix = np.random.randint(0,10,size=(5,5))
# print(matrix)

# import tensorflow as tf
# input = tf.constant(np.random.rand(3,4))
# k = 2
# output = tf.nn.top_k(input,k)
# with tf.Session() as sess:
#     print(sess.run(input))
#     print(sess.run(output))

def foo(x,*args):
    print(x)
    print(args)

if __name__ == '__main__':
    matrix1 = np.array([[1,0,3,0],
             [0,0,0,1]])
    matrix1 = sparse.lil_matrix(matrix1)
    print(matrix1.rows[0])
    # print(foo(1,2,3,4,5))
    # row = np.array([0, 0, 1, 2, 2, 2])
    # col = np.array([0, 2, 2, 0, 1, 2])
    # data = np.array([1, 2, 3, 4, 5, 6])
    # mtx = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
    # # print(mtx)
    # # print()
    # # print(mtx.todok())
    # print(mtx)
    # print(mtx.todense())
    # print(mtx.getrow(1).nonzero()[1])
    # print(mtx.getrow(1).toarray()[0])