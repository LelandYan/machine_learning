# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/9 22:32'

import os
import re

file_list = []
path = r"C:\Users\lenovo\Desktop\my_notes"


def file_split(filename):
    try:
        md = filename.split('.')[1]
    except Exception as e:
        md = ""
    return md


def traversal(path):
    fs = os.listdir(path)
    for f in fs:
        temp_path = os.path.join(path, f)
        if not os.path.isdir(temp_path) and file_split(f) == "md":
            print("文件" + temp_path)
            file_list.append(temp_path)
        else:
            try:
                traversal(temp_path)
            except Exception as e:
                print("无法解析" + temp_path)
                return


def handle():
    traversal(path)
    for file in file_list:
        newLines = ""
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                if line.__contains__("###") == True:
                    newLine = line.split("###")
                    newLines = newLines + ("###" + " " + newLine[1])
                else:
                    newLines = newLines + line
        with open(file, 'w', encoding='utf8') as f:
            f.write(newLines)
if __name__ == '__main__':
    handle()
