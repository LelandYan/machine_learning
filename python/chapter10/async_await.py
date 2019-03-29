# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/29 13:10'

async def downloader(url):
    return "bobby"

async def download_url(url):
    html = await downloader(url)
    return html

if __name__ == '__main__':
    coro = download_url("http://www.baidu.com")
    coro.send(None)



