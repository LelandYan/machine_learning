# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/26 7:27'

from random import randint, seed

seed(1)
data = [randint(-10, 10) for _ in range(10)]

res = filter(lambda x: x >= 0, data)
res2 = [x for x in data if x >= 0]

from collections import namedtuple

Student = namedtuple("Student", ["name", "age"])
s = Student("Jim", 16)

c = dict.fromkeys(data, 0)

from collections import Counter

c2 = Counter(data)
# print(c2.most_common(3))

# print(sorted([1,2,3,3]))
d = {x: randint(60, 100) for x in "xyzabc"}

# print(list(zip(d.values(),d.keys())))
# print(sorted(d.items(),key=lambda x:x[1]))
from collections import OrderedDict
from collections import deque
import pickle
# 可迭代器的切片问题
from itertools import islice
# 串行访问多个迭代器
from itertools import chain
# 并行访问多个迭代器
# zip([12,2],[1,2])
# 分隔用多个分隔符的字符串
# s1 = "123 123:\sdfsd; sdfsdf;\dsf"
# import re
# rest = re.split(r"[,:;\\ ]",s1)
# rset = [x for x in rest if x]
# print(rset)

import os,stat

# s = "123"
# print(s.ljust(20,'='))

# # 解码
# import struct
# struct.unpack()

#访问文件的大小
import os,stat
# s = os.stat("deque_learning.py")
# 获取文件类型
# print(s.st_mode)
print(dir(os.path))