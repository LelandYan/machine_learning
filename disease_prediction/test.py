# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/17 20:29'

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import model

CSV_FILE_PATH = 'csv_result-colonTumor.csv'
df = pd.read_csv(CSV_FILE_PATH)
shapes = df.values.shape
data = df.values[:, 1:shapes[1] - 1]
result = df.values[:, shapes[1] - 1:shapes[1]]
value_len = data.shape[1]
pop_len = result.shape[0]

DNA_SIZE = value_len  # DNA length
POP_SIZE = pop_len  # population size
CROSS_RATE = 0.8  # mating probability(DNA crossover)
MUTATION_RATE = 0.003  # mutation probability
N_GENERATIONS = 10


def translateDNA(pop):
    # return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    # select_value = pop.astype(np.bool)
    # for i in select_value:
    #     data[i] = data[select_value]
    # return data
    # cnt = 0
    # if pop[0] == 1:
    #     pop[0] = 0
    #     cnt += 1
    # for i in range(len(pop)):
    #     if i != 0:
    index_list = []
    for i in range(len(pop)):
        if pop[i] == 1:
            index_list.append(i)
    print("index:", index_list)
    return index_list


pop = np.zeros((POP_SIZE, DNA_SIZE))
pop = np.full(pop.shape, 0)
count = 1
for i in range(len(pop)):
    for j in range(len(pop[i])):
        if count > 0.005 * DNA_SIZE:
            if np.random.rand() < 0.8:
                pop[i][j] = 1
                count += 1
for _ in range(N_GENERATIONS):
    accuracy_list = []
    feature_list = []
    for i in range(data.shape[0]):
        data = data[:, [1,2,45,35]]
        print(data)
