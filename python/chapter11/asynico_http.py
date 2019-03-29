# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/29 17:33'

# asyncio 没有提供http协议的接口 aiohttp可以使用
import asyncio
import socket
from urllib.parse import urlparse


async def get_url(url):
    # 通过socket请求html
    url = urlparse(url)
    host = url.netloc
    path = url.path
    if path == "":
        path = "/"

    # 建立socket连接
    reader, writer = await asyncio.open_connection(host, 80)
    send_path = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
    writer.write(send_path.encode("utf8"))
    all_lines = []
    async for raw_line in reader:
        data = raw_line.decode("utf8")
        all_lines.append(data)
    html = "\n".join(all_lines)
    return html


async def main():
    tasks = []
    for url in range(20):
        new_url = "http://shop.projectsedu.com/goods/{}/".format(url)
        tasks.append(asyncio.ensure_future(get_url(new_url)))
    for task in asyncio.as_completed(tasks):
        result = await  task
        print(result)


if __name__ == '__main__':
    import time

    start_time = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    print("last time:{}".format(time.time() - start_time))
