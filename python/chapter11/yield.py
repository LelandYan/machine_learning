# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/30 12:29'
import random


# 使用yield关键字，将普通的函数变成生成器
def mygen(alist):
    while len(alist) > 0:
        c = random.randint(0, len(alist) - 1)
        yield alist.pop(c)


a = ["aa", "bb", "cc"]
c = mygen(a)


# 生成器就是一个迭代器，可以使用for进行跌打，生成器的最大的特点就是可以接受传入的一个变量，并根据变量的内容计算结果后返回
def gen():
    value = 0
    while True:
        receive = yield value
        if receive == "e":
            break
        value = f"got {receive}"


# g = gen()
# print(g.send(None))
# print(g.send("hello"))
# print(g.send(123456))
# print(g.send('e'))


def g1():
    yield range(5)


def g2():
    yield from range(5)


it1 = g1()
it2 = g2()
# for x in it1:
#     print(x)
# for x in it2:
#     print(x)

"""
yield将range这个可跌打对象直接返回了
yield from 解析了range对象，将其中的每一个item返回了
yield from iterable == for item in iterable:yield item
"""


def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n += 1


def f_wrapper(fun_iterable):
    print("start")
    for item in fun_iterable:
        yield item
    print("end")


# wrap = f_wrapper(fab(5))
# for i in wrap:
#     print(i,end=" ")

import logging


def f_wrapper2(fun_iterable):
    print("start")
    yield from fun_iterable
    print("end")


# wrap2 = f_wrapper2(fab(5))
# print(wrap2)
# for i in wrap2:
#     print(i,end=" ")

import asyncio, random


@asyncio.coroutine
def smart_fib(n):
    index = 0
    a = 0
    b = 1
    while index < n:
        sleep_secs = random.uniform(0, 0.2)
        yield from asyncio.sleep(sleep_secs)
        print(f"Smart one think {sleep_secs} secs to get {b}")
        a, b = b, a + b
        index += 1


@asyncio.coroutine
def stupid_fib(n):
    index = 0
    a = 0
    b = 1
    while index < n:
        sleep_secs = random.uniform(0, 0.4)
        yield from asyncio.sleep(sleep_secs)
        print(f"Stupid one think {sleep_secs} secs to get {b}")
        a, b = b, a + b
        index += 1


# if __name__ == '__main__':
#     loop = asyncio.get_event_loop()
#     tasks = [
#         smart_fib(10),
#         stupid_fib(10),
#     ]
#     loop.run_until_complete(asyncio.wait(tasks))
#     print("All fib finished")
#     loop.close()

# def inner_generator():
#     i = 0
#     while True:
#         i = yield i
#         if i > 10:
#             raise StopIteration
#
# def outer_generator():
#     print("do something before yield")
#     from_inner = 0
#     from_outer = 1
#     g = inner_generator()
#     g.send(None)
#     while True:
#         try:
#             from_inner = g.send(from_outer)
#             from_outer = yield from_inner
#         except StopIteration:
#             break
#
# def main():
#     g = outer_generator()
#     g.send(None)
#     i = 0
#     while True:
#         try:
#             i = g.send(i+1)
#             print(i)
#         except StopIteration:
#             break

#
#
# if __name__ == '__main__':
#     main()

# @asyncio.coroutine
# def countdown(number,n):
#     while n < 0:
#         print("T-minus",n,'({})'.format(number))
#         yield from asyncio.sleep(1)
#         n -= 1
# loop = asyncio.get_event_loop()
# tasks = [
#     asyncio.ensure_future(countdown('A',2)),
#     asyncio.ensure_future(countdown('B',2))
# ]
# loop.run_until_complete(asyncio.wait(tasks))
# loop.close()


import time
import requests


async def wait_download(url):
    await download(url)
    print(f"get {url} data complete")


async def download(url):
    response = requests.get(url)
    print(response.text)


async def main():
    start = time.time()
    await asyncio.wait([
        wait_download("http://www.163.com"),
        wait_download("http://www.mi.com"),
        wait_download("http://www.baidu.com"),
        wait_download("https://blog.csdn.net/u013205877/article/details/70502508"),
        wait_download("https://blog.csdn.net/u013205877/article/details/70502503"),
        wait_download("https://blog.csdn.net/u013205877/article/details/70502502"),
    ])
    end = time.time()
    print(f"Complete in {end-start} seconds")


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
