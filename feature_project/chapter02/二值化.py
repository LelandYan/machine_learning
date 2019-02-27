# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/27 20:06'

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the data about businesses
biz_file = open('yelp_academic_dataset_business.json')
biz_df = pd.DataFrame([json.loads(x) for x in biz_file])
biz_file.close()

# sns.set_style("whitegrid")
# fig, ax = plt.subplots()
# biz_df['review_count'].hist(ax=ax, bins=100)
# ax.set_yscale("log")
# ax.tick_params(labelsize=14)
# ax.set_xlabel("Review Count", fontsize=14)
# ax.set_ylabel("Occurrence", fontsize=14)
# plt.show()
# 分位数装箱
# 绘制中位数以及图像
# deciles = biz_df['review_count'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])
# sns.set_style("whitegrid")
# fig, ax = plt.subplots()
# biz_df['review_count'].hist(ax=ax, bins=100)
# for pos in deciles:
#     handle = plt.axvline(pos, color='r')
# ax.legend([handle],['deciles'],fontsize=14)
# ax.set_yscale("log")
# ax.set_xscale("log")
# ax.tick_params(labelsize=14)
# ax.set_xlabel("Review Count", fontsize=14)
# ax.set_ylabel("Occurrence", fontsize=14)
# plt.show()

# 固定宽度的箱进行量化计数
# small_count = np.random.randint(0, 100, 20)
# np.floor_divide(small_count, 10)
# large_counts = [296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 11495, 91897, 44, 28, 7971, 926, 122, 22222]
# print(np.floor(np.log10(large_counts)))
# print(pd.qcut(large_counts,4,labels=False))
# large_counts_series = pd.Series(large_counts)
# print(large_counts_series.quantile([0.25,0.5,0.75]))

# 对数转化
# fig,(ax1,ax2) = plt.subplots(2,1)
# biz_df['review_count'].hist(ax=ax1,bins=100)
# ax1.tick_params(labelsize=14)
# ax1.set_xlabel("review_count",fontsize=14)
# ax1.set_ylabel("Occurrence",fontsize=14)
#
# np.log(biz_df['review_count']).hist(ax=ax2,bins=100)
# ax2.tick_params(labelsize=14)
# ax2.set_xlabel("log10(review_count)",fontsize=14)
# ax2.set_ylabel("Occurrence",fontsize=14)
# plt.show()


# 对数转化实战
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


biz_df['log_review_count'] = np.log10(biz_df['review_count'] + 1)
m_orig = linear_model.LinearRegression()
scores_orig = cross_val_score(m_orig,biz_df[['review_count']],biz_df['stars'],cv=10)
print(scores_orig)
m_log = linear_model.LinearRegression()
scores_log = cross_val_score(m_orig,biz_df[['log_review_count']],biz_df['stars'],cv=10)
print(scores_log)

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.scatter(biz_df['review_count'], biz_df['stars'])
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Review Count', fontsize=14)
ax1.set_ylabel('Average Star Rating', fontsize=14)

ax2.scatter(biz_df['log_review_count'], biz_df['stars'])
ax2.tick_params(labelsize=14)
ax2.set_xlabel('Log of Review Count', fontsize=14)
ax2.set_ylabel('Average Star Rating', fontsize=14)
plt.show()