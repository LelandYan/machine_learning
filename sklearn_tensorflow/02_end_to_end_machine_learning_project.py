# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/1 16:50'

import tarfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib

HOUSING_PATH = "datasets/housing"


def fetch_housing_data(housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    housing_tgz = tarfile.open("housing.tgz")
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# split the data first method
def split_train_test(data, test_ratio):
    # 设置随机种子,以产生总是相同的洗牌指数,防止多次运行后,你会得到整个数据集
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    # iloc[i]是获取第i行数据
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


if __name__ == '__main__':
    housing = load_housing_data()
    # print(housing.shape)
    # 查看前五行
    # print(housing.head())

    # 快速查看数据的描述，特别是总行数，每个属性的类型和非空值的数量
    # print(housing.info())

    # 每个类别中包含了多少个街区
    # print(housing['ocean_proximity'].value_counts())

    # 显示的是数值属性的cout,mean,min,max扽
    # print(housing.describe())

    # 可视化
    # housing.hist(bins=50,figsize=(20,15))
    # plt.show()

    # split the data second method

    # adds an 'index' column
    # housing_with_id = housing.reset_index()
    # # train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"index")
    #
    # housing_with_id['id'] = housing["longitude"] * 1000 + housing["latitude"]
    # train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"id")

    from sklearn.model_selection import train_test_split

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    """后面的代码通过将收入中位数除以 1.5（以限制收入分类的数量），
    创建了一个收入类别属性，用ceil对值舍入（以产生离散的分类），
    然后将所有大于 5的分类归入到分类 5："""

    housing['income_cat'] = np.ceil(housing["median_income"] / 1.5)
    # inplace = True 不创建新的对象，直接对原始的对象进行修改
    # inplace = False 对数据进行修改，创建并返回新的对象承载其修改结果
    print(housing['income_cat'])
    housing['income_cat'].where(housing["income_cat"] < 5, 5.0, inplace=True)
    print(housing['income_cat'])
    # 根据收入分类，进行分类采样，可以使用StratifiedShuffleSplit
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # print(housing["income_cat"].value_counts() / len(housing))

    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)

    # 数据的探索和可视化，发展规律
    # 创建一个副本，以免损伤训练集
    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"] / 100,
                 label="population",
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    # plt.legend()
    # plt.show()

    # 查找关联 使用corr()方法，计算出每对属性间的标准相关系数 standard correlation coefficient 皮尔逊相关系数
    # corr_matrix = housing.corr()
    # print(corr_matrix["median_house_value"].sort_values(ascending=False))
    from pandas.tools.plotting import scatter_matrix

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    corr_matrix = housing.corr()
    # print(corr_matrix["median_house_value"].sort_values(ascending=True))

    # 创建干净的数据集
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()


    # 处理缺失值的方法
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy="median")

    # 注意这个数据不能存在非数值的数据，创建一个不包含文本属性的数据副本
    housing_num = housing.drop("ocean_proximity",axis=1)
    imputer.fit(housing_num)
    # 返回的实例变量statistics_列出了每个属性的中位数
    print(imputer.statistics_==housing_num.median().values)

    # 将缺失值转化为中位数，返回的是一个numpy的数组
    X = imputer.transform(housing_num)

    # 将其转为Pandas的DataFrame中
    housing_tr = pd.DataFrame(X,columns=housing_num.columns)

    # 处理文本和类别属性
    from sklearn.preprocessing import LabelEncoder
    # encoder = LabelEncoder()
    housing_cat = housing["ocean_proximity"]
    # housing_cat_encoded = encoder.fit_transform(housing_cat)
    # print(housing_cat_encoded)

    # 具有多个文本特征列的时候
    housing_cat_encoded,housing_categories = housing_cat.factorize()


    # 独热编码 One-Hot-Encoding
    from sklearn.preprocessing import OneHotEncoder
    # encoder = OneHotEncoder()
    # housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
    # print(housing_cat_1hot.toarray())

    from sklearn.preprocessing import LabelBinarizer
    # 向构造器LabelBinarizer中传入sparse_output=True 就可以得到一个稀疏矩阵
    encoder = LabelBinarizer()
    housing_cat_1hot = encoder.fit_transform(housing_cat)
    # print(housing_cat_1hot)

