# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/10 20:10'

import threading


# class XiaoAi(threading.Thread):
#     def __init__(self, lock):
#         super().__init__(name="小爱同学")
#         self.lock = lock
#
#     def run(self):
#         self.lock.acquire()
#         print(f"{self.name} : 在")
#         self.lock.release()
#
#         self.lock.acquire()
#         print(f"{self.name} : 好啊")
#         self.lock.release()
#
#
# class TianMao(threading.Thread):
#     def __init__(self, lock):
#         super().__init__(name="天猫精灵")
#         self.lock = lock
#
#     def run(self):
#         self.lock.acquire()
#         print(f"{self.name} : 小爱同学")
#         self.lock.release()
#
#         self.lock.acquire()
#         print(f"{self.name} : 对古诗")
#         self.lock.release()

from threading import Condition
class XiaoAi(threading.Thread):
    def __init__(self, cond):
        super().__init__(name="小爱同学")
        self.cond = cond

    def run(self):
        with self.cond:
            self.cond.wait()
            print(f"{self.name} : 在")
            self.cond.notify()

            self.cond.wait()
            print(f"{self.name} : 好啊")
            self.cond.notify()


class TianMao(threading.Thread):
    def __init__(self, cond):
        super().__init__(name="天猫精灵")
        self.cond = cond

    def run(self):
        with self.cond:
            print(f"{self.name} : 小爱同学")
            self.cond.notify()
            self.cond.wait()

            print(f"{self.name} : 对古诗")
            self.cond.notify()
            self.cond.wait()

if __name__ == '__main__':
    cond = threading.Condition()
    xiaoai = XiaoAi(cond)
    tianmao = TianMao(cond)


    xiaoai.start()
    tianmao.start()
