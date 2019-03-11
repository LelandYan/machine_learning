# _*_ coding: utf-8 _*_

from random import randrange,choice
from string import ascii_lowercase as lc
from sys import maxsize
from time import ctime

tlds = ("com","edu","net","org","gov")

for i in range(randrange(5,11)):
    # pick data
    dtint =  randrange(maxsize)
    # data string
    dtstr = ctime(dtint)
    # login is shorter
    llen = randrange(4,8)
    login = ''.join(choice(lc for k in range(llen)))
    dlen = randrange(llen,13)
    dom = ''.join(choice(lc) for j in range(dlen))
    print(dtstr,"::",login,"@",dom,".",choice(tlds),"::",dtint,"-",llen,"-",dlen)
# print(randrange(5,11))