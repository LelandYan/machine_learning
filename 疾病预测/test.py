# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/17 20:29'

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

CSV_FILE_PATH = 'csv_result-ALL-AML_train.csv'
df = pd.read_csv(CSV_FILE_PATH)
shapes = df.values.shape
data = df.values[:, 1:shapes[1] - 1]
result = df.values[:, shapes[1] - 1:shapes[1]]
train_x, test_x, train_y, test_y = train_test_split(data, result, test_size=0.3)
print(np.array(train_x.flatten()).shape)
