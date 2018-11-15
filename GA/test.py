import numpy as np
import matplotlib.pyplot as plt

pop = np.random.randint(2, size=(100, 10))
#print(pop)
#print(2 ** np.arange(10)[::-1])
#print((2 ** np.arange(10)[::-1] / float(2 ** 10 - 1) * 5).shape)
#print(float(2 ** 10 - 1) * 5)
res = pop.dot(2 ** np.arange(10)[::-1] / float(2 ** 10 - 1) * 5)
#print(res)
#print([np.random.permutation(10) for _ in  range(100)])
# print(np.random.normal(1,0.2))
print(np.random.rand(10))