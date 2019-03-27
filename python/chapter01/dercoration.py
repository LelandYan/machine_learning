# # _*_ coding: utf-8 _*_
# __author__ = 'LelandYan'
# __date__ = '2019/3/27 12:45'
#
# from functools import wraps
# from threading import Thread
#
# import dis
#
# # def add(a):
# #     a += 1
# #     return a
#
# # print(dis.dis(add))
#
# total = 0
#
#
# def add():
#     global total
#     for i in range(100000):
#         total += 1
#
#
# def desc():
#     global total
#     for i in range(100000):
#         total -= 1
#
#
# # import threading
# # threading.Thread(target=add).start()
# # threading.Thread(target=desc).start()
# # print(total)
# import time
# import threading
#
#
# def get_detail_html(url):
#     print("get detail html started")
#     time.sleep(2)
#     print("get detail html end")
#
#
# def get_detail_url(url):
#     print("get detail url started")
#     time.sleep(2)
#     print("get detail url end")
#
#
# class get_detail_url_class(threading.Thread):
#     def __init__(self, name):
#         super().__init__(name=name)
#
#     def run(self):
#         print("get detail url started")
#         time.sleep(2)
#         print("get detail url end")
#
#
# if __name__ == '__main__':
#     thread1 = threading.Thread(target=get_detail_html, args=("",))
#     thread2 = threading.Thread(target=get_detail_url, args=("",))
#     start_time = time.time()
#     thread1.start()
#     thread2.start()
#     thread1.join()
#     thread2.join()
#     print(f"last time: {time.time()-start_time}")


import time
import threading
from queue import Queue

def get_detail_html(queue):
    while True:
        url = queue.get()
        print("get detail html started")
        time.sleep(2)
        print("get detail html end")


def get_detail_url(queue):
    while True:
        print("get detail url started")
        time.sleep(2)
        for i in range(20):
            queue.put(f"http://www.as.com/{i}")
        print("get detail url end")

if __name__ == '__main__':
    detail_url_queue = Queue(maxsize=1000)
    thread1 = threading.Thread(target=get_detail_html, args=(detail_url_queue,))
    thread2 = threading.Thread(target=get_detail_url, args=(detail_url_queue,))
    start_time = time.time()
    thread1.start()
    thread2.start()
    # 主线程等待thread1执行完后再执行主线程
    thread1.join()
    thread2.join()
    print(f"last time: {time.time()-start_time}")

