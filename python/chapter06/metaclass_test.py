# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/9 17:32'


# 用type创建动态类
def say(self):
    return self.name
    # return "i am user"


class BaseClass:
    def answer(self):
        return "i am baseclass"


Users = type("User", (BaseClass,), {"name": "user", "say": say})
my_obj = Users()


# print(my_obj.name)
# print(my_obj.say())
# print(my_obj.answer())


# 元类 创建类的类  对象<-class（对象）<-type
class MetaClass(type):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


class User2(metaclass=MetaClass):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "user"


if __name__ == '__main__':
    my_obj2 = User2(name="bobby")
    print(my_obj2)
