# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/1 15:22'

import matplotlib.pyplot as plt
import sklearn.linear_model
import matplotlib as mpl
import pandas as pd
import numpy as np
from six.moves import urllib


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli['INEQUALITY'] == 'TOT']
    oecd_bli = oecd_bli.pivot(index='Country',columns='Indicator',values='Value')
    gdp_per_capita.rename(columns={'2015':"GDP per capita"},inplace=True)
    gdp_per_capita.set_index("Country",inplace=True)
    full_country_stats = pd.merge(left=oecd_bli,right=gdp_per_capita,left_index=True,right_index=True)
    full_country_stats.sort_values(by='GDP per capita',inplace=True)
    remove_indices = [0,1,6,8,33,34,35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[['GDP per capita','Life satisfaction']].iloc[keep_indices]



# 修改matplotlib的默认参数
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# load data
# thousands 每千的分割符 na_values 填充缺失值
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')
# prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats['Life satisfaction']]
# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()