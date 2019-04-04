# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/4 15:37'

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv",thousands=',')
gdp_per_capital = pd.read_csv("gdp_per_capital.csv",thousands=",",delimiter='\t',encoding="latin1",na_values="n/a")


# prepare the data
def prepare_country_stats(oecd_bli,gdp_per_capital):
    # get the pandas data_frame of GDP per capital Life satisfaction
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=='TOT']
    oecd_bli = oecd_bli.pivot_table(index='Country',column="Indicator",values="Value")
        


if __name__ == '__main__':
    df_test = pd.DataFrame({'foo': ['one','one','one','two','two','two'],
                       'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                       'baz': [1, 2, 3, 4, 5, 6]})
    res = df_test.pivot(index="foo",columns='bar',values='baz')
    print(res)