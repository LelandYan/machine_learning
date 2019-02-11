# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/9 15:25'
from collections import namedtuple,defaultdict
from collections.abc import *

# 1. tuple 不可变的
name_tuple = ['bobby1','bobby2','bobby3','bobby4']
name_list = ['bobby1','bobby2']
# __iter__ __getitem__ 可迭代的
# 拆包
# name1,name2 = name_tuple
name3,*other = name_tuple
# 可作为dict的key，不可变可哈希，list不可以作为key


########################################

def gen_default():

    return {
        "name":"",
        "nums":0
    }
default_dict2 = defaultdict(gen_default)
default_dict = defaultdict(int)
users = ['bobby1','bobby2','bobby3','bobby1','bobby2']
for user in users:
    default_dict[user] += 1
print(defaultdict)