# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/22 9:14'

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.stats as ss
import numpy as np
import seaborn as sns

sns.set_context(font_scale=1.5)
# df = pd.read_csv("./data/HR.csv")
# sns.set_style(style="darkgrid")
# sns.set_context(context="poster",font_scale=0.8)
# sns.set_palette("summer")
# f= plt.figure()
# f.add_subplot(131)
# sns.distplot(df["satisfaction_level"],bins=10,)


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

# 交叉分析方法
# df = pd.read_csv("./data/HR.csv")
# dp_indices = df.groupby(by="department").indices
# sales_values = df["left"].iloc[dp_indices["sales"]].values
# technical_values = df["left"].iloc[dp_indices["technical"]].values
# # print(ss.ttest_ind(sales_values,technical_values)[1])
# dp_keys = list(dp_indices.keys())
# dp_t_mat = np.zeros([len(dp_keys),len(dp_keys)])
# for i in range(len(dp_keys)):
#     for j in range(len(dp_keys)):
#         p_value = ss.ttest_ind(df["left"].iloc[dp_indices[dp_keys[i]]].values,df["left"].iloc[dp_indices[dp_keys[j]]].values)[1]
#         if p_value <0.05:
#             dp_t_mat[i][j] = -1
#         else:
#             dp_t_mat[i][j] = p_value
# sns.heatmap(dp_t_mat,xticklabels=dp_keys,yticklabels=dp_keys)
# plt.show()

# 透视表
# df = pd.read_csv("./data/HR.csv")
# piv_tb = pd.pivot_table(df,values="left",index=["promotion_last_5years","salary"],columns=["Work_accident"],\
#                         aggfunc=np.mean)
# sns.heatmap(piv_tb,vmin=0,vmax=1)
# plt.show()


# 分组分析
sns.set_context(font_scale=1.5)
# df = pd.read_csv("./data/HR.csv")
# sns.barplot(x="salary",y="left",hue="department",data=df)
# plt.show()
# sl_s = df["satisfaction_level"]
# sns.barplot(list(range(len(sl_s))),sl_s.sort_values())
# plt.show()

# 相关分析
# sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap=sns.color_palette("RdBu",n_colors=128))
# plt.show()

# 因子分析
# from sklearn.decomposition import PCA
# my_pca = PCA(n_components=7)
# lower_mat = my_pca.fit_transform(df.drop(labels=["salary","department","left"],axis=1))
# print("Ratio:",my_pca.explained_variance_ratio_)
# sns.heatmap(pd.DataFrame(lower_mat).corr(),vmin=-1,vmax=1,cmap=sns.color_palette("RdBu",n_colors=128))
# plt.show()

from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel

# df = pd.DataFrame({'A':ss.norm.rvs(size=10),'B':ss.norm.rvs(size=10),'C':ss.norm.rvs(size=10),"D":np.random.randint(low=0,high=2,size=10)})
# print(df)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# X = df.loc[:, ["A", "B", "C"]]
# Y = df.loc[:, "D"]

# 过滤思想
# skb = SelectKBest(k=2)
# skb.fit(X,Y)
# skb.transform(X)
#
# # ;包裹思想
# rfe = RFE(estimator=SVR(kernel="linear"),n_features_to_select=2,step=1)
# rfe.fit_transform(X,Y)
#
# # 嵌入思想
# sfm = SelectFromModel(estimator=DecisionTreeRegressor(),threshold=0.1) # threshold 设置的阀值
# sfm.fit_transform(X,Y)


from sklearn.preprocessing import Normalizer
# 正规化是对行进行正规化
# Normalizer(norm="l1").fit_transform(np.array([[1,1,3]]))

# print(np.array([1,1,3]).shape)
# print(np.array([1,1,3]).reshape(-1,1).shape)
# print(np.array([[1,1,3]]).shape)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

def hr_preprocessing(sl=False, le=False, npr=False, amh=False, tsc=False, wa=False, pl2=False, dp=False, slr=False,lower_d=False,ld_n=1):
    """
    :param sl:satisfaction_level  False - MinMaxScaler True --- StandardScaler
    :param le: last_evaluation  False - MinMaxScaler True --- StandardScaler
    :param npr number_project --- False - MinMaxScaler True --- StandardScaler
    :param amh average_monthly_hours --- False - MinMaxScaler True --- StandardScaler
    :param tsc time_spend_company --- False - MinMaxScaler True --- StandardScaler
    :param wa Work_accident --- False - MinMaxScaler True --- StandardScaler
    :param pl2 promotion_last_5years --- False - MinMaxScaler True --- StandardScaler
    :param dp department --- False:LabelEncoding True:OneHotEncoding
    :param slr salary  --- False:LabelEncoding True:OneHotEncoding
    :param ower_d is or not pca
    :param ld_n= dim
    :return:
    """
    df = pd.read_csv("./data/HR.csv")
    # 清洗数据 (去除异常值(空值))
    df = df.dropna(subset=["satisfaction_level", "last_evaluation"])
    df = df[df["satisfaction_level"] <= 1][df["salary"] != "nme"]
    # 得到标注
    label = df["left"]
    # 指定以列进行删除
    df = df.drop("left", axis=1)
    # 特征选择

    # 特征处理
    scaler_list = [sl, le, npr, amh, tsc, wa, pl2]
    column_list = ["satisfaction_level", "last_evaluation", "number_project", "average_monthly_hours",
                   "time_spend_company", "Work_accident", "promotion_last_5years"]
    for i in range(len(scaler_list)):
        if not scaler_list[i]:
            df[column_list[i]] = MinMaxScaler().fit_transform(df[column_list[i]].values.reshape(-1, 1)).reshape(1, -1)[
                0]
        else:
            df[column_list[i]] = \
                StandardScaler().fit_transform(df[column_list[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

    scaler_lst = [slr, dp]
    column_lst = ["salary", "department"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i] == "salary":
                df[column_lst[i]] = [map_salary(s) for s in df["salary"].values]
            else:
                df[column_lst[i]] = LabelEncoder().fit_transform(df[column_lst[i]])
        else:
            # 对DataFrame进行one-hot编码的时候，用processing比较麻烦
            df = pd.get_dummies(df, columns=[column_lst[i]])
        df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_list[i]].values.reshape(-1, 1)).reshape(1, -1)[
            0]
    if lower_d:
        return PCA(n_components=ld_n).fit_transform(df.values),label
    return df,label


def map_salary(s):
    d = dict([('low', 0), ('medium', 1), ("hight", 2)])
    return d.get(s, 0)


def main():
    print(hr_preprocessing(lower_d=True))


if __name__ == '__main__':
    main()
