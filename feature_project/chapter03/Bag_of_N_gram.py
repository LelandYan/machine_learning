# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/28 8:03'

# Bag-of-N-gram

import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer

X_test = ['I sed about sed the lack','of any Actually']
# stop=words=None 表示去掉停用词，若改为stop_words='english' 则去掉停用词
count_vec = CountVectorizer(stop_words=None)
# 稀疏矩阵的表示方式
print(count_vec.fit_transform(X_test))
# 表示的统计的词频结果
print(count_vec.fit_transform(X_test).toarray())

print("\nvocabulary list:\n\n",count_vec.vocabulary_)



# f = open("yelp_academic_dataset_review.json")
# js = []
# for i in range(1000):
#     js.append(json.loads(f.readline()))
# f.close()
#
# review_df = pd.DataFrame(js)
"""
(?u)放在前面表示对匹配中的大小写不明显
紧接着的\b"和末尾的"\b"表示匹配两个词语的间隔(可以简单的理解为空格)
中间的"\w"表示匹配一个字母或数字或下划线或汉字，紧接着的"\w+“表示匹配一个或者多个字母或数字或下划线或汉字
所以这个正则表达式就会忽略掉单个的字符，因为它不满足这个正则表达式的匹配，因此我们可以给他改为”(?u)\b\w+\b"，这样就不会忽略单个的字符。
"""
# ngram_range：tuple (min_n, max_n) 要提取的不同n-gram的n值范围的下边界和上边界。 将使用n的所有值，使得min_n <= n <= max_n
# bow_converter = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
# bigram_converter = CountVectorizer(ngram_range=(2,2), token_pattern='(?u)\\b\\w+\\b')
# trigram_converter = CountVectorizer(ngram_range=(3,3), token_pattern='(?u)\\b\\w+\\b')