# _*_ coding: utf-8 _*_
import pandas as pd



data = pd.DataFrame({"id":[1,2,3],"age":[21,22,23]})
# print(data)
cov_data = pd.Series([data[c].value_counts().index[0] for c in data],index=data.columns)
print(cov_data)
# pd.Series([X[c].value_counts().index[0] for c in X],
#                                         index=X.columns)
# print(data[c].value_counts().index[0] for c in data)
# for c in data:
#     print(data[c].value_counts().index[0])