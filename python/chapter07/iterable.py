# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/9 21:50'


def fib(index):
    if index <= 2:
        return 1
    else:
        return fib(index - 1) + fib(index - 2)
