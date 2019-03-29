# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/29 10:03'

# 通过select实现http请求
import socket
from urllib.parse import urlparse
from selectors import DefaultSelector, EVENT_READ, EVENT_WRITE

selector = DefaultSelector()
urls = ["http://www.baidu.com"]
stop = False
class Fetcher:
    def readable(self, key):
        d = self.client.recv(1024)
        if d:
            self.data += d
        else:
            selector.unregister(key.fd)
            data = self.data.decode("utf8")
            html_data = data.split("\r\n\r\n")[1]
            print(html_data)
            self.client.close()
            urls.remove(self.spider_url)
            if not urls:
                global stop
                stop = True
    def connected(self, key):
        selector.unregister(key.fd)
        send_path = f"GET {self.path} HTTP/1.1\r\nHost: {self.host}\r\nConnection: close\r\n\r\n".encode("utf8")
        self.client.send(send_path)

        selector.register(self.client.fileno(), EVENT_READ, self.readable)

    def get_url(self, url):
        self.spider_url = url
        url = urlparse(url)
        self.host = url.netloc
        self.path = url.path
        self.data = b""
        if self.path == "":
            self.path = "/"

        # 建立socket连接
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.setblocking(False)
        try:
            self.client.connect((self.host, 80))
        except BlockingIOError as e:
            pass
        # 注册
        selector.register(self.client.fileno(), EVENT_WRITE, self.connected)

def loop():
    # select本身是不支持register模型的
    # socket状态变化的以后的回调有程序院完成
    while not stop:
        ready = selector.select()
        for key,mask in ready:
            call_back = key.data
            call_back(key)


# def get_url(url):
#     # 通过socket请求html
#     url = urlparse(url)
#     host = url.netloc
#     path = url.path
#     if path == "":
#         path = "/"
#
#     # 建立socket连接
#     client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client.setblocking(False)
#     # 堵塞不会消耗cpu
#     try:
#         client.connect((host, 80))
#     except BlockingIOError as e:
#         pass
#     send_path = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
#     while True:
#         try:
#             client.send(send_path.encode("utf8"))
#             break
#         except OSError as e:
#             pass
#     # client.send(send_path.encode("utf8"))
#     data = b""
#     while True:
#         try:
#             d = client.recv(1024)
#         except BlockingIOError as e:
#             continue
#         if d:
#             data += d
#         else:
#             break
#     data = data.decode("utf8")
#     html_data = data.split("\r\n\r\n")[1]
#     print(html_data)
#     client.close()
#

if __name__ == '__main__':
    # get_url("http://www.baidu.com")
    fetcher = Fetcher()
    fetcher.get_url("http://www.baidu.com")
    loop()
