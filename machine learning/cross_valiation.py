# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/1 10:52'

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer,accuracy_score

data = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(data['data'],data['target'],test_size=0.2,random_state=0)
# print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
dec_clf = DecisionTreeClassifier(random_state=0)
dec_clf.fit(X_train,y_train)
# print(dec_clf.score(X_test,y_test))

parameters = {'max_depth':range(1,6)}
# 构建一个打分器
scoring_fnc = make_scorer(accuracy_score)
kfold = KFold(n_splits=10)


# for train_index,test_index in kfold.split(range(10)):
#     print(train_index,test_index)

# 网格搜索
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(dec_clf,parameters,scoring_fnc,cv=KFold)
# grid.fit(X_train,y_train)
# print(grid.best_estimator_)
# print(grid.best_score_)
# print(grid.score(X_test,y_test))
