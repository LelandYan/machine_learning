# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/31 7:46'

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 导入数据
data = datasets.load_iris()
col_name = data.feature_names
X = data.data
y = data.target

# print(col_name)

# 数据预览
# X = pd.DataFrame(X)
# print(X.head(n=10))
# print(X.sample(n=10))
# print(X.shape)
# print(X.dtypes)
# print(X.describe())


# 数据可视化
# 通过箱线图可以发现数据隐藏的异常值
# X.plot(kind='box')
# plt.show()

# 通过条形图可以看出数据结构的分布，是否满足正太分布
# X.hist(figsize=(12, 5), xlabelsize=1, ylabelsize=1)
# plt.show()

# 通过折线图可以知道数据值大小的密度分布情况
# X.plot(kind="density", subplots=True, layout=(4, 4), figsize=(12, 5))
# plt.show()

# 通过特征相关图我们能够知道哪些特征存在明显的相关性
# pd.scatter_matrix(X, figsize=(10, 10))
# plt.show()

# 和之前的特征相关图相比，热力图更加清晰跟个特征之间的关系
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# cax = ax.matshow(X.corr(), vmin=-1, vmax=1, interpolation="none")
# fig.colorbar(cax)
# ticks = np.arange(0,4,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(col_name)
# ax.set_yticklabels(col_name)
# plt.show()

# 查找最优模型
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler

models = []
models.append(("AB", AdaBoostClassifier()))
models.append(("GBM", GradientBoostingClassifier()))
models.append(("RF", RandomForestClassifier()))
models.append(("ET", ExtraTreesClassifier()))
models.append(("SVC", SVC()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("LR", LogisticRegression()))
models.append(("GNB", GaussianNB()))
models.append(("LDA", LinearDiscriminantAnalysis()))

names = []
results = []

for name, model in models:
    kfold = KFold(n_splits=5, random_state=42)
    result = cross_val_score(model, X, y, scoring='accuracy', cv=kfold)
    names.append(name)
    results.append(result)
    print("{} Mean:{:.4f}(Std:{:.4f})".format(name, result.mean(), result.std()))


print()
# 使用Pipeline
from sklearn.pipeline import Pipeline

pipeline = []
pipeline.append(("ScalerET", Pipeline([("Scaler",StandardScaler()),
 ("ET",ExtraTreesClassifier())])))
pipeline.append(("ScalerGBM", Pipeline([("Scaler",StandardScaler()),
   ("GBM",GradientBoostingClassifier())])))
pipeline.append(("ScalerRF", Pipeline([("Scaler",StandardScaler()),
 ("RF",RandomForestClassifier())])))

names = []
results = []

for name, model in pipeline:
    kfold = KFold(n_splits=5, random_state=42)
    result = cross_val_score(model, X, y, scoring='accuracy', cv=kfold)
    names.append(name)
    results.append(result)
    print("{} Mean:{:.4f}(Std:{:.4f})".format(name, result.mean(), result.std()))

print()
# 模型的调节
param_grid = {
    "C":[0.1,0.3,0.5,0.7,0.9,1.0,1.3,1.5,1.7,2.0],
    "kernel":["linear","poly","rbf","sigmoid"]
}
model = SVC()
kfold = KFold(n_splits=5,random_state=42)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring="accuracy",cv=kfold)
grid_result = grid.fit(X,y)
print("Best: {} using {}".format(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean,stdev,param in zip(means,stds,params):
    print("{} ({}) with {}".format(mean,stdev,param))