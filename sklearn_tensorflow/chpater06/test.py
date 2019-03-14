# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/14 22:18'
import numpy as np

np.random.seed(21)
names = ["肖一卓", "刘丽珍", "崔晨旭", "李蒙", "欧大啸", "张雪倩", "靳羽希", "吴清典","徐冰冰"]
a = np.random.permutation(range(1,16))
b = np.random.permutation(range(1,16))
c = np.random.permutation(range(1,16))
item_list = []
for i1,i2,i3 in zip(a,b,c):
    if(i1 != i2 and i2 != i3 and i1 != i3):
        item_list.append((i1,i2,i3))

for name,item in zip(names,item_list):
    print(name,item)