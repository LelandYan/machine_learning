# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/29 13:30'

# 事件循环+回调（驱动生成器）+epoll（IO多路复用）
# import asyncio
# import time
# async def get_html(url):
#     print("start get url")
#     await asyncio.sleep(2)
#     print("end get url")
#
#
# if __name__ == '__main__':
#     start_time = time.time()
#     loop = asyncio.get_event_loop()
#     task = [get_html(i) for i in range(100)]
#     loop.run_until_complete(asyncio.wait(task))
#     print(time.time()-start_time)
#

# 获取协程的返回值
import asyncio
import time


async def get_html(url):
    print("start get url")
    await asyncio.sleep(2)
    return "123"


if __name__ == '__main__':
    start_time = time.time()
    loop = asyncio.get_event_loop()
    get_future = asyncio.ensure_future(get_html("http"))
    task = [get_html(i) for i in range(100)]
    loop.run_until_complete(asyncio.wait(task))
    print(time.time() - start_time)
