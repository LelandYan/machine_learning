# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/14 12:48'

# 通过非阻塞io实现http请求
import socket
from urllib.parse import urlparse


def get_url(url):
    # 通过socket请求html
    url = urlparse(url)
    host = url.netloc
    path = url.path
    if path == "":
        path = "/"

    # 建立socket连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setblocking(False)
    # 堵塞不会消耗cpu
    try:
        client.connect((host, 80))
    except BlockingIOError as e:
        pass
    send_path = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
    while True:
        try:
            client.send(send_path.encode("utf8"))
            break
        except OSError as e:
            pass
    # client.send(send_path.encode("utf8"))
    data = b""
    while True:
        try:
            d = client.recv(1024)
        except BlockingIOError as e:
            continue
        if d:
            data += d
        else:
            break
    data = data.decode("utf8")
    html_data = data.split("\r\n\r\n")[1]
    print(html_data)
    client.close()

if __name__ == '__main__':
    get_url("http://www.baidu.com")