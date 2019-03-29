# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/29 10:46'

from itertools import chain
my_list = [1,2,3]
my_dict = {
    "test1":1,
    "test2":2
}
for value in chain(my_list,my_dict,range(5,10)):
    print(value)