# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/28 9:44'

import pandas as pd
from sklearn import linear_model

df = pd.DataFrame({'City': ['SF', 'SF', 'SF', 'NYC', 'NYC', 'NYC', 'Seattle', 'Seattle', 'Seattle'],
                   'Rent': [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]})

one_hot_df = pd.get_dummies(df, prefix=['city'])
model = linear_model.LinearRegression()
model.fit(one_hot_df[['city_NYC','city_SF','city_Seattle']],one_hot_df[['Rent']])
# 斜率
print(model.coef_)
# 截距
print(model.intercept_)
