# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/8 17:43'

# is 比较的id
# == 比较的值是否相等
# instance(a,b) 判断的类型
# 主要是使用instance而不是type

class Date:
    """
    日期
    """
    def __init__(self,birthday):
        # 私有属性
        self.__birthday = birthday

    # 实例方法
    def tomorrow(self):
        pass

    # 静态方法
    @staticmethod
    def parse_from_string(date_string):
        return Date(123)

    # 类方法
    @classmethod
    def from_string(cls,date_string):
        return cls(123)
print(Date.__dict__)
print(dir(Date))
print()