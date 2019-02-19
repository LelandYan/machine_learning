# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/14 18:10'

from collections import namedtuple, defaultdict


class Node:
    def __init__(self, value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

    def depth_first(self):
        yield self
        for c in self:
            yield from c.depth_first()


def frange(start, stop, increment):
    x = start
    while x < stop:
        yield x
        x += increment


def countdown(n):
    print("start to count from", n)
    while n > 0:
        yield n
        n -= 1
    print("Done")


class Node2:
    def __init__(self, value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

    def depth_first(self):
        return DepthFirstIterator(self)


class DepthFirstIterator:
    def __init__(self, start_node):
        self._node = start_node
        self._children_iter = None
        self._child_iter = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._children_iter is None:
            self._children_iter = iter(self._node)
            return self._node
        elif self._child_iter:
            try:
                nextchild = next(self._child_iter)
                return nextchild
            except StopIteration:
                self._child_iter = None
                return next(self)
        else:
            self._child_iter = next(self._children_iter).depth_first()
            return next(self)


class CountDown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1

    def __reversed__(self):
        n = 1
        while n <= self.start:
            yield n
            n += 1


from collections import deque


class linehistroy:
    def __init__(self, lines, histlen=3):
        self.lines = lines
        self.history = deque(maxlen=histlen)

    def __iter__(self):
        for lineno, line in enumerate(self.lines, 1):
            self.history.append((lineno, line))
            yield line

    def clear(self):
        self.history.clear()


if __name__ == '__main__':
    pass
    with open("somefile.txt") as f:
        lines = linehistroy(f)
        for line in lines:
            if 'python' in line:
                for lineno,hline in lines.history:
                    print("{}:{}".format(lineno,hline),end=' ')
    # for rr in reversed(CountDown(10)):
    #     print(rr)
    # for rr in CountDown(10):
    #     print(rr)
    # # c = countdown(3)
    # for i in c:
    #     print(i)
    # for n in frange(0, 4, 0.5):
    #     print(n)
    # print(list(frange(0, 4, 0.5)))
    # root = Node2(0)
    # child1 = Node2(1)
    # child2 = Node2(2)
    # root.add_child(child1)
    # root.add_child(child2)
    # child1.add_child(Node2(3))
    # child1.add_child(Node2(4))
    # child2.add_child(Node2(5))
    # for ch in root.depth_first():
    #     print(ch)
    # root = Node(1)
    # root.add_child(Node(1))
    # root.add_child(Node(1))
    # for ch in root:
    #     print(ch)
    # for i in reversed([1,2,3]):
    #     print(i)
