# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/19 22:18'


class random_pratice:
    def __init__(self, name=None, each_problems=3):
        self.name = name
        self.name_num = len(self.name)
        self.each_problems = each_problems
        self.file_num = None

    def shuffle(self):
        from random import seed, sample
        seed(42)
        self._item_list = []
        for i in range(self.name_num):
            each_one = sample(self.file_list, self.each_problems)
            self._item_list.append(each_one)

    def produce_item(self):
        self.name_item = {}
        for name, item in zip(self.name, self._item_list):
            self.name_item[name] = item

    def produce_file(self):
        from shutil import copy
        from os.path import join
        for item in self.name_item.keys():
            value = self.name_item.get(item)
            for i in value:
                i = str(i) + ".md"
                file_from_path = join(self.abs_path, i)
                file_to_path = join(self.abs_path, item, i)
                copy(file_from_path, file_to_path)

    def file_counter(self):
        from os import getcwd, listdir, path
        self.abs_path = getcwd()
        cnt = [name for name in listdir(self.abs_path) if
               path.isfile(path.join(self.abs_path, name)) and name.endswith("md")]
        self.file_list = [int(i.split('.')[0]) for i in cnt if i.split('.')[0].isdigit()]
        self.file_num = len(self.file_list)

    def make_dir(self):
        self.item_path = []
        from os import path, makedirs
        from shutil import rmtree
        for name in self.name_item.keys():
            item_path = path.join(self.abs_path, name)
            self.item_path.append(item_path)
            if path.exists(item_path):
                rmtree(item_path)
            makedirs(item_path)

    def run(self):
        self.file_counter()
        self.shuffle()
        self.produce_item()
        self.make_dir()
        self.produce_file()


if __name__ == '__main__':
    a = random_pratice(name=["肖一卓", "刘丽珍", "崔晨旭", "李蒙", "欧大啸", "张雪倩", "靳羽希", "吴清典", "徐冰冰"], each_problems=3)
    a.run()
