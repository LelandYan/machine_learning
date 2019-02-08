# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/8 17:34'

import abc
from collections.abc import *

class CacheBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get(self, key):
        pass

    @abc.abstractmethod
    def set(self, key, value):
        pass


class RedisCache(CacheBase):
    pass

redis_cache = RedisCache()