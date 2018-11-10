# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/10 23:43'
import matplotlib.pyplot as plt
import numpy as np

# DNA size
N_CITIES = 20
CROSS_RATE =  0.1
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 500

class GA(object):
    def __init__(self,DNA_size,cross_rate,mutation_rate,pop_size):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in  range(pop_size)])
    # get cities cord in order
    def translateDNA(self,DNA,city_position):
        line_x = np.empty_like(DNA,dtype=np.float64)
        line_y = np.empty_like(DNA,dtype=np.float64)
        for i,d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i,:] = city_coord[:,0]
            line_y[i,:] = city_coord[:,1]
        return line_x,line_y