# _*_ coding: utf-8 _*_

# 导入数据
import numpy as np
from numpy.random import choice
from sklearn.datasets import load_iris

# 特征矩阵加工
iris = load_iris()
iris.data = np.hstack((choice([0, 1, 2], size=iris.data.shape[0] + 1).reshape(-1, 1),
                       np.vstack((iris.data, np.array([np.nan, np.nan, np.nan, np.nan]).reshape(1, -1)))))
iris.target = np.hstack((iris.target,np.array([np.median(iris.target)])))


# 并行处理
#
from numpy import log
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import FeatureUnion

# 将新建整体特征矩阵进行对数函数转化对象
step2_1 = ("ToLog",FunctionTransformer(log))
# 将新建的整体特征矩阵进行二值化类对对象
setp2_2 = ("ToBinary",Binarizer())
# 对新建整体并行处理
step2 = ("FeatureUnion",FeatureUnion(transformer_list=[step2_1,setp2_2]))

#