# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/19 23:55'


# def randint(start, end, seed=99999):
#     a = 32310901
#     b = 1729
#     rOld = seed
#     m = end - start
#     while True:
#         rNew = (a * rOld + b) % m
#         yield rNew
#         rOld = rNew


# for _ in range(20):
#     rnd = randint(1,10)
#     for _ in range(10):
#         print(next(rnd))
m = 2 ** 32
a = 1103515245
c = 12345
rdls = []
def LCG(seed,mi,ma,n):
    if n == 1:
        return 0
    else:
        seed = (a*seed+c)%m
        rdls.append(int((ma-mi)*seed/float(m-1))+mi)
        print(int((ma-mi)*seed/float(m-1))+mi)
        LCG(seed,mi,ma,n-1)

def main():
    mi = 0
    ma = 15
    co = 10
    import time
    # seed = time.time()-1
    seed = 23 * (10**8)
    print(seed)
    LCG(seed,mi,ma,co)
    print(set(rdls))

class test:
    def __init__(self):
        self.rdls = []
        self.m = 2 ** 32
        self.c = 12345
        self.a = 1103515245

    def gen_random(self, mi, ma, co, seed):
        seed = seed * (10 ** 8)
        self.LCG(seed, mi, ma, co * 3)
        return self.rdls[:co]

    def LCG(self, seed, mi, ma, n):
        if n == 1:
            return 0
        else:
            seed = (self.a * seed + self.c) % self.m
            self.rdls.append(int((ma-mi)*seed/float(self.m-1))+mi)
            self.LCG(seed, mi, ma, n - 1)




if __name__ == '__main__':
    a = [1,2,3]
    print(a[::3])





# def LCG(seed):
#     seed = (a * seed + c) % m
#     return seed /float(m-1)
#
# def main():
#     m1 = 1
#     m2 = 15
#     import time
#     seed = time.time()
#     print(seed)
#     rd = LCG(seed)
#     ourd = int((m2-m1)*rd) + m1
#     print(ourd)
# if __name__ == '__main__':
#     main()
#     main()
