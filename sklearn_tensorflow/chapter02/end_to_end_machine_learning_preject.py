# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/5 11:51'

import numpy as np
import os

# to make output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figure
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)


def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id, '.', fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


import warnings

warnings.filterwarnings(action='ignore', message='internal gelsd')

import pandas as pd

HOUSING_PATH = os.path.join('.', 'datasets')


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


housing = load_housing_data()


# print(housing.info())
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())

# housing.hist(bins=50,figsize=(20,15))

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set)/len(test_set))


from sklearn.model_selection import train_test_split

# housing["income_cat"] = np.ceil(housing['median_income'] / 1.5)
# housing['income_cat'].where(housing['income_cat'] < 5,5.0,inplace=True)
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

# print(housing['income_cat'].value_counts())
# housing['income_cat'].hist()
# plt.show()


# 分层抽样
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
compare_props = pd.DataFrame({
    'Overall': income_cat_proportions(housing),
    'Stratified': income_cat_proportions(strat_test_set),
    'Random': income_cat_proportions(test_set)
})
compare_props['Rand. %error'] = 100 * compare_props['Random'] / compare_props['Overall'] - 100
compare_props['Strat. %error'] = 100 * compare_props['Stratified'] / compare_props['Overall'] - 100

# print(compare_props)

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#     s=housing["population"]/100, label="population", figsize=(10,7),
#     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#     sharex=False)
# plt.legend()
# plt.show()

corr_matrix = housing.corr()
corr = corr_matrix['median_house_value'].sort_values(ascending=True)
# print(corr)

# from pandas.plotting import scatter_matrix
#
# attributes = ["median_house_value", "median_income", "total_rooms",
#               "housing_median_age"]
# scatter_matrix(housing[attributes],figsize=(12,8))


# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#              alpha=0.1)
# plt.axis([0, 16, 0, 550000])
# plt.show()

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr = corr_matrix["median_house_value"].sort_values(ascending=False)
# print(corr)


housing = strat_train_set.drop('median_house_value', axis=1)
housing_lables = strat_train_set["median_house_value"].copy()

# 取出值为nan的行
sample_incomplete_row = housing[housing.isnull().any(axis=1)].head()
# print(sample_incomplete_row)

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))
print(housing_tr.loc[sample_incomplete_row.index.values])

housing_tr = pd.DataFrame(X, columns=housing_num.columns)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

num_attributes = list(housing_num)
cat_attribute = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attributes)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
from CategoricalEncoder import CategoricalEncoder
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribute)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared.shape)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_lables)


