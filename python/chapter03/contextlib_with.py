# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/8 20:19'

import contextlib

@contextlib.contextmanager
def file_open(file_name):
    print("file open")
    yield {}
    print("file end")


with file_open("a.txt") as f_opened:
    print("file processing")