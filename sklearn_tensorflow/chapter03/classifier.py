# _*_ coding: utf-8 _*_
from sklearn.datasets import fetch_mldata
from scipy.sparse import lil_matrix
import numpy as np
user_item_matrix = lil_matrix((4,4))
user_item_matrix[0,1] = 1
user_item_matrix[0,2] = 2
user_item_matrix[0,0] = 3
user_item_matrix[1,2] = 2
print(user_item_matrix.todense())
print(np.array(user_item_matrix.nonzero()).T)
user_to_positive_set = {u: set(row) for u, row in enumerate(user_item_matrix.rows)}
print(user_to_positive_set)