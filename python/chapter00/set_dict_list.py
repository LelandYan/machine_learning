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

import os, stat

# s = "123"
# print(s.ljust(20,'='))

# # 解码
# import struct
# struct.unpack()

# 访问文件的大小
import os, stat
# s = os.stat("deque_learning.py")
# 获取文件类型
# print(s.st_mode)
# print(dir(os.path))

# 创建临时文件对象
from tempfile import TemporaryFile, NamedTemporaryFile

from xml.etree.ElementTree import parse


class IntTuple(tuple):
    def __new__(cls, iterable):
        g = (x for x in iterable if isinstance(x, int) and x > 0)
        return super(IntTuple, cls).__new__(cls, g)
    # def __init__(self,iterable):
    #     super(IntTuple,self).__init__(iterable)


class Player:
    def __init__(self, uid, name, status=0, level=1):
        self.uid = uid
        self.name = name
        self.stat = status
        self.level = level

from functools import total_ordering
from abc import ABCMeta,abstractmethod
# __slots__ = [] 阻止动态绑定属性
@total_ordering
class Player2:
    __slots__ = ["uid", "name", "stat", "level"]

    def __init__(self, uid, name, status=0, level=1):
        self.uid = uid
        self.name = name
        self.stat = status
        self.level = level
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def getUid(self):
        return self.uid
    def setUid(self,uid):
        self.uid = uid
    R = property(getUid,setUid)
import telnetlib
if __name__ == '__main__':

    # t = IntTuple([1,-1,'abc',6,['x','y'],3])
    # print(t)
    pass
    Player("001",'Jim')
    import weakref
    a_wref = weakref.ref(1)
    from threading import Thread
    import Queue

