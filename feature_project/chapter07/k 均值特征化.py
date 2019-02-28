# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/28 16:59'

# import numpy as np
# from sklearn.cluster import KMeans
#
#
# class KMeansFeaturizer:
#
#     def __init__(self, k=100, target_scale=5.0, random_state=None):
#         """将数字型数据输入K-均值聚类
#         在输入数据上运行k-均值并把每个数据点设定为它的团id，如果存在目标变量，则将其缩放并包含为k-均值的输入以导出服从分类边界以及组相似点的簇
#         """
#         self.k = k
#         self.target_scale = target_scale
#         self.random_state = random_state
#
#     def fit(self, X, y=None):
#         """在输入的数据上运行k均值，并找到中心"""
#         if y is None:
#             km_model = KMeans(n_clusters=self.k, n_init=20, random_state=self.random_state)
#             km_model.fit(X)
#             self.km_model_ = km_model
#             self.cluster_centers_ = km_model.cluster_centers_
#             return self
#
#         # 若有目标信息，使用合适的缩减并把输入的数据k均值
#         data_with_target = np.hstack((X, y[:, np.newaxis] * self.target_scale))
#
#         # 在数据和目标上建立预测训练k-均值模型
#         km_model_pretrain = KMeans(n_clusters=self.k, n_init=20, random_state=self.random_state)
#         km_model_pretrain.fit(data_with_target)
#
#         # 运行k-均值第二次获得团在原始空间没有目标信息，使用预先训练中发信质心进行初始化
#         # 通过一个迭代的集群分配和质心重新计算
#         km_model = KMeans(n_clusters=self.k, init=km_model_pretrain.cluster_centers_[:, :2], n_init=1, max_iter=1)
#         km_model.fit(X)
#         self.km_model = km_model
#         self.cluster_centers_ = km_model.cluster_centers_
#         return self
#
#     def transform(self, X, y=None):
#         """为每个输入数据点输出最接近团id"""
#         clusters = self.km_model.predict(X)
#         return clusters[:, np.newaxis]
#
#     def fit_transform(self, X, y=None):
#         self.fit(X, y)
#         return self.transform(X, y)
#
#
# from scipy.spatial import SphericalVoronoi, voronoi_plot_2d
# from sklearn.datasets import make_moons
#
# training_data, training_labels = make_moons(n_samples=200, noise=0.2)
# kmf_hint = KMeansFeaturizer(k=100, target_scale=10).fit(training_data, training_labels)
# kmf_no_hint = KMeansFeaturizer(k=100, target_scale=0).fit(training_data, training_labels)
#
#
# def kmeans_voronoi_plot(X, y, cluster_center, ax):
#     """绘制与数据叠加的k-均值类的Voronoi图"""
#     ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", alpha=0.2)
#     vor = SphericalVoronoi(cluster_center)
#     voronoi_plot_2d(vor, ax=ax, show_vertices=False, alpha=0.5)
#
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from scipy import sparse
#
# # 生成与训练数据相同的测试数据
# test_data, test_labels = make_moons(n_samples=2000, noise=0.3)
#
# # 生成k均值特技生成特征
# training_cluster_features = kmf_hint.transform(training_data)
# test_cluster_features = kmf_hint.transform(test_data)
#
# # 将新的输入特征和聚类特征整合
# training_with_cluster = sparse.hstack((training_data, training_cluster_features))
# test_with_cluster = sparse.hstack((test_data, test_cluster_features))
#
# seed = 666
# # 建立分类器
# lr_cluster = LogisticRegression(random_state=seed).fit(training_with_cluster, training_labels)
# classifier_names = ['LR', 'kNN', 'RBF SVM', 'Random Forest', 'Boosted Trees']
# classifiers = [LogisticRegression(random_state=seed), KNeighborsClassifier(5), SVC(gamma=2, C=1),
#                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#                GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=5)]
# for model in classifiers:
#     model.fit(training_data, training_labels)
#
# import sklearn
#
#
# # 辅助函数使用ROC评估分类器性能
# def test_roc(model, data, labels):
#     if hasattr(model, "decision_function"):
#         predictions = model.decision_function(data)
#     else:
#         predictions = model.predict_proba(data)[:, 1]
#         fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
#         return fpr, tpr
#
#
# # 显示结果
# import matplotlib.pyplot as plt
#
# plt.figure()
#
# fpr_cluster, tpr_cluster = test_roc(lr_cluster, test_with_cluster, test_labels)
# plt.plot(fpr_cluster, tpr_cluster, 'r-', label='LR with k-means')
#
# for i, model in enumerate(classifiers):
#     fpr, tpr = test_roc(model, test_data, test_labels)
#     plt.plot(fpr, tpr, label=classifier_names[i])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.legend()
