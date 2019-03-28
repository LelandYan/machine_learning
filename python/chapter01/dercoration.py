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


# import time
# import threading
# from queue import Queue
#
# def get_detail_html(queue):
#     while True:
#         url = queue.get()
#         print("get detail html started")
#         time.sleep(2)
#         print("get detail html end")
#
#
# def get_detail_url(queue):
#     while True:
#         print("get detail url started")
#         time.sleep(2)
#         for i in range(20):
#             queue.put(f"http://www.as.com/{i}")
#         print("get detail url end")
#
# if __name__ == '__main__':
#     detail_url_queue = Queue(maxsize=1000)
#     thread1 = threading.Thread(target=get_detail_html, args=(detail_url_queue,))
#     thread2 = threading.Thread(target=get_detail_url, args=(detail_url_queue,))
#     start_time = time.time()
#     thread1.start()
#     thread2.start()
#     # 主线程等待thread1执行完后再执行主线程
#     thread1.join()
#     thread2.join()
#     print(f"last time: {time.time()-start_time}")

# import threading
# import time
#
#
# class HtmlSpider(threading.Thread):
#     def __init__(self, url,sem):
#         super().__init__()
#         self.url = url
#         self.sem = sem
#
#     def run(self):
#         time.sleep(2)
#         print("got html text success")
#         self.sem.release()
#
# class UrlProducer(threading.Thread):
#     def __init__(self, sem):
#         super().__init__()
#         self.sem = sem
#
#     def run(self):
#         for i in range(20):
#             self.sem.acquire()
#             html_thread = HtmlSpider(f"http://baidu.com/{i}",self.sem)
#             html_thread.start()
#
#
# if __name__ == '__main__':
#     sem = threading.Semaphore(3)
#     url_producer = UrlProducer(sem)
#     url_producer.start()

# import time
# from concurrent.futures import ThreadPoolExecutor,as_completed
#
#
# def get_html(times):
#     time.sleep(times)
#     print(f"got page {times} success")
#     return times
#
#
# executor = ThreadPoolExecutor(max_workers=2)
# urls = [2,3,4]
# all_task = [executor.submit(get_html,(url)) for url in urls]
# # for future in as_completed(all_task):
#     data = future.result()
#     print(f"get {data} page success")
# for data in executor.map(get_html,urls):
#     print(f"get {data} page success")

# task1 = executor.submit(get_html, (3))
# task2 = executor.submit(get_html, (2))
#
# # done 用于判定某个任务是或完成 非阻塞
# print(task1.done())
# # 当在执行中获取执行完，是不能执行的
# print(task2.cancel())
# # 获取返回的结果
# print(task1.result())
import dis
a  = 1
def fun():
    global  a
    a += 1

print(dis.dis(a))

