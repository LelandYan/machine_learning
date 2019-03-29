# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/14 16:54'

import asyncio
import time
from functools import partial

# async def get_html(url):
#     print("start get url")
#     await asyncio.sleep(2)
#     return "bobby"
#
#
# def callback(url,future):
#     print(url)
#     print("send email to bobby")
#
#
# if __name__ == '__main__':
#     start_time = time.time()
#     loop = asyncio.get_event_loop()
#     # get_future = asyncio.ensure_future(get_html("http://www.baidu.com"))
#     task = loop.create_task(get_html("http://www.baidu.com"))
#     task.add_done_callback(partial(callback,"http:"))
#     # tasks = [get_html("http://www.baidu.com") for i in range(100)]
#     loop.run_until_complete(task)
#     # print(time.time()-start_time)
#     print(task.result())
