# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/28 16:59'

import numpy as np
from sklearn.cluster import KMeans


class KMeansFeaturizer:

    def __init__(self,k=100,target_scale=5.0,random_state=None):
        """将数字型数据输入K-均值聚类
        在输入数据上运行k-均值并把每个数据点设定为它的团id，如果存在目标变量，则将其缩放并包含为k-均值的输入以导出服从分类边界以及组相似点的簇
        """
        self.k = k
        self.target_scale = target_scale
        self.random_state = random_state

    def fit(self,X,y=None):
        """在输入的数据上运行k均值，并找到中心"""
        if y is None:
            km_model = KMeans(n_clusters=self.k,n_init=20,random_state=self.random_state)
            km_model.fit(X)
            self.km_model_ = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self

        # 若有目标信息，使用合适的缩减并把输入的数据k均值
        data_with_target = np.hstack((X,y[:,np.newaxis]*self.target_scale))

        # 在数据和目标上建立预测训练k-均值模型
        km_model_pretrain = KMeans(n_clusters=self.k,n_init=20,random_state=self.random_state)
        km_model_pretrain.fit(data_with_target)

        # 运行k-均值第二次获得团在原始空间没有目标信息，使用预先训练中发信质心进行初始化
        # 通过一个迭代的集群分配和质心重新计算
        km_model = KMeans(n_clusters=self.k, init=km_model_pretrain.cluster_centers_[:, :2], n_init=1, max_iter=1)
        km_model.fit(X)
        self.km_model = km_model
        self.cluster_centers_ = km_model.cluster_centers_
        return self

    def transform(self,X,y=None):
        """为每个输入数据点输出最接近团id"""
        clusters = self.km_model.predict(X)
        return clusters[:,np.newaxis]

    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X,y)


