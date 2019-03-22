# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/22 9:14'

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("./data/HR.csv")
sns.set_style(style="darkgrid")
sns.set_context(context="poster",font_scale=0.8)
sns.set_palette("summer")
f= plt.figure()
f.add_subplot(131)
sns.distplot(df["satisfaction_level"],bins=10,)


# sns.countplot(x="salary",data=df,hue="department")
# plt.show()
# plt.title("SALARY")
# plt.xlabel("salary")
# plt.ylabel("Number")
# plt.xticks(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts().index)
# plt.axis([0,4,0,10000])
# plt.bar(np.arange(len(df["salary"].value_counts()))+0.5, df["salary"].value_counts(),width=0.5)
# for x,y in zip(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts()):
#     plt.text(x,y,y,ha="center",va="bottom")
# plt.show()
