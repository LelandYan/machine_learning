# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/25 13:17'

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def hr_preprocessing(sl=False, le=False, npr=False, amh=False, tsc=False, wa=False, pl2=False, dp=False, slr=False,
                     lower_d=False, ld_n=1):
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
        return PCA(n_components=ld_n).fit_transform(df.values), label
    return df, label


def map_salary(s):
    d = dict([('low', 0), ('medium', 1), ("hight", 2)])
    return d.get(s, 0)


def hr_modeling(features, label):
    from sklearn.model_selection import train_test_split
    f_v = features
    f_names = features.columns.values
    l_v = label
    X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)
    # print(len(X_train),len(X_validation),len(X_test))

    # KNK
    from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    from sklearn.naive_bayes import GaussianNB,BernoulliNB
    from sklearn.tree import DecisionTreeClassifier,export_graphviz
    import pydotplus
    from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.externals.six import StringIO
    # BernoulliNB 适合离散二值的
    # GaussianNB
    models = []
    models.append(("KNN",KNeighborsClassifier(n_neighbors=3)))
    models.append(("GaussianNB",GaussianNB()))
    models.append(("BerouliNB",BernoulliNB()))
    models.append(("DecisionTreeGini",DecisionTreeClassifier()))
    models.append(("DecisionTreeEntropy", DecisionTreeClassifier(criterion="entropy")))
    models.append(("SVC",SVC(kernel='rbf',C=1000)))
    models.append(("RandomForest",RandomForestClassifier()))
    models.append(("AdaBoostClassifier",AdaBoostClassifier(n_estimators=100)))
    models.append(("LogisticRegression",LogisticRegression(C=1000,tol=1e-10,solver="sag",max_iter=10000)))
    models.append(("GradientBoostingClassifier",GradientBoostingClassifier(max_depth=6,n_estimators=100)))
    for clf_name,clf in models:
        clf.fit(X_train,Y_train)
        xy_lst = [(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(i)
            print(clf_name,"ACC:",accuracy_score(Y_pred,Y_part))
            print(clf_name, "REC:", recall_score(Y_pred, Y_part))
            print(clf_name, "F1:", f1_score(Y_pred, Y_part))
            # dot_data = export_graphviz(clf,out_file=None,
            #                            feature_names=f_names,
            #                            class_names=["NL","L"],
            #                            filled=True,
            #                            special_characters=True,
            #                            rounded=True)
            # graph = pydotplus.graph_from_dot_data(dot_data)
            # graph.write_pdf("dt_tree.pdf")
def repr_test(features,label):
    # print(features)
    # print(label)
    from sklearn.linear_model import LinearRegression,Ridge,Lasso
    regr = LinearRegression()
    # regr = Ridge(alpha=1)
    # regr = Lasso(alpha=0.2)
    regr.fit(features.values,label.values)
    Y_pred = regr.predict(features.values)
    print(regr.coef_)
    from sklearn.metrics import mean_squared_error
    print("MSE:",mean_squared_error(Y_pred,label.values))

def main():
    features, label = hr_preprocessing()
    # repr_test(features[["number_project","average_monthly_hours"]],features["last_evaluation"])
    hr_modeling(features, label)


if __name__ == '__main__':
    main()
