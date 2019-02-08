# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/8 20:30'

from collections import abc
import numbers
import bisect
a = [3, 4, 5, 6, 7, 9, 11, 13, 15, 17]


# 返回列表
# print(a[::])
# # 逆向输出
# print(a[::-1])

class Group:
    def __init__(self, group_name, company_name, staffs):
        self.group_name = group_name
        self.company_name = company_name
        self.staffs = staffs

    def __reversed__(self):
        self.staffs.reverse()

    def __getitem__(self, item):
        cls = type(self)
        if isinstance(item, slice):
            return cls(self.group_name, self.company_name, self.staffs[item])
        if isinstance(item, numbers.Integral):
            return cls(self.group_name, self.company_name, [self.staffs[item]])

    def __iter__(self):
        return iter(self.staffs)

    def __len__(self):
        return len(self.staffs)

    def __contains__(self, item):
        if item in self.staffs:
            return True
        else:
            return False

from collections import deque
inter_list = []
# 处理已经排序的序列
# 采用二分法，效率较高
bisect.insort(inter_list,3)


import array
