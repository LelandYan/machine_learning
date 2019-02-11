# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/9 16:43'
from datetime import date,datetime

class User:
    def __init__(self,name,birthday):
        self.name = name
        self.birthday = birthday
        self._age = 0

    @property
    def age(self):
        return datetime.now().year - self.birthday.year

    @age.setter
    def age(self,value):
        self._age = value

    def __getattr__(self, item):
        pass

    def __getattribute__(self, item):
        pass

if __name__ == '__main__':
    user = User("bobby",date(1987,1,1))
    print(user.age)
