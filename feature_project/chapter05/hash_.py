# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/28 11:00'

import pandas as pd
import json
from sklearn.feature_extraction import FeatureHasher


# js = []
# with open('yelp_academic_dataset_review.json') as f:
#     for i in range(10000):
#         js.append(json.loads(f.readline()))
#
#
# review_df = pd.DataFrame(js)
# print(js)
# m = len(review_df.business_id.unique())
# h = FeatureHasher(n_features=m, input_type='string')
# f = h.transform(review_df['business_id'])
# print(f.toarray())
# print('Our pandas Series, in bytes:')