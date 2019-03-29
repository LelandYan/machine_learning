# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/29 17:22'

import asyncio
import socket
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
def get_url(url):
    # 通过socket请求html
    url = urlparse(url)
    host = url.netloc
    path = url.path
    if path == "":
        path = "/"

    # 建立socket连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 堵塞不会消耗cpu
    client.connect((host, 80))
    send_path = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
    client.send(send_path.encode("utf8"))
    data = b""
    while True:
        d = client.recv(1024)
        if d:
            data += d
        else:
            break
    data = data.decode("utf8")
    html_data = data.split("\r\n\r\n")[1]
    print(html_data)
    client.close()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()
    task = loop.run_in_executor(executor,get_url,"http://www.baidu.com")
    loop.run_until_complete(asyncio.wait(task))