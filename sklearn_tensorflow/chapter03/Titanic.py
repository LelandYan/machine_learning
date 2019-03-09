# _*_ coding: utf-8 _*_

import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

TITANIC_PATH = os.path.join("datasets", "titanic")


def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


if __name__ == '__main__':
    train_data = load_titanic_data("train.csv")
    test_data = load_titanic_data("test.csv")
    # print(train_data.head())
    # print(train_data.info())
    # print(train_data.describe())
    # print(train_data["Survived"].value_counts())

    num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", Imputer(strategy="median"))
    ])
    # print(num_pipeline.fit_transform(train_data))
    cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
    # print(cat_pipeline.fit_transform(train_data).shape)
    preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    X_train = preprocess_pipeline.fit_transform(train_data)
    y_train = train_data["Survived"]

    svm_clf = SVC(gamma="auto")
    svm_clf.fit(X_train, y_train)

    X_test = preprocess_pipeline.fit_transform(test_data)
    y_pred = svm_clf.predict(X_test)

    svm_scores = cross_val_score(svm_clf,X_train,y_train,cv=10)
    print(svm_scores.mean())

    forest_clf = RandomForestClassifier(n_estimators=100,random_state=42)
    forest_scores = cross_val_score(forest_clf,X_train,y_train,cv=10)
    print(cross_val_score(forest_clf,X_train,y_train).mean())
    # result = forest_clf.predict(X_test)
    # file = open("gender_submission.csv","w")
    # file.write("PassengerId,Survived\n")
    # cnt = 892
    # for i in result:
    #     file.write(str(cnt)+","+str(i)+"\n")
    #     cnt += 1

