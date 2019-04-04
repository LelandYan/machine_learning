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
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capital.rename(columns={"2015":'GDP per capital'},inplace=True)
    gdp_per_capital.set_index("Country",inplace=True)
    full_country_stats = pd.merge(left=oecd_bli,right=gdp_per_capital,left_index=True,right_index=True)
    return full_country_stats[["GDP per capital","Life satisfaction"]]

country_stats = prepare_country_stats(oecd_bli,gdp_per_capital)
country_stats.to_csv("country_stats.csv",encoding='utf-8')
X = np.c_[country_stats["GDP per capital"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind="scatter",x='GDP per capital',y='Life satisfaction')

# Select a linear model
lin_reg_model = LinearRegression()

# Train the model
lin_reg_model.fit(X,y)

# plot Regression model
t0,t1 = lin_reg_model.intercept_[0],lin_reg_model.coef_[0][0]
X = np.linspace(0, 110000, 1000)
plt.plot(X,t0+t1*X,'k')
plt.show()












