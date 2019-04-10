# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/4/10 13:46'

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv("spam.csv",encoding="latin-1")
df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
df['label'] = df['class'].map({'ham':0,'spam':1})
X = df["message"]
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X)
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=1/3,random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))
