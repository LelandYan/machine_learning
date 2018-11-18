import numpy as np
import matplotlib.pyplot as plt
import random

print(random.randint(0, 1))
pop = np.random.randint(2, size=(100, 10))
# print(pop)
# print(2 ** np.arange(10)[::-1])
# print((2 ** np.arange(10)[::-1] / float(2 ** 10 - 1) * 5).shape)
# print(float(2 ** 10 - 1) * 5)
res = pop.dot(2 ** np.arange(10)[::-1] / float(2 ** 10 - 1) * 5)
# print(res)
# print([np.random.permutation(10) for _ in  range(100)])
# print(np.random.normal(1,0.2))
# print(np.random.rand(10))
a = [1, 2, 3, 4, 5, 6]
print(a[1:6:2])
