# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/17 23:02'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import model

CSV_FILE_PATH = 'csv_result-ALL-AML_train.csv'
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
N_GENERATIONS = 200
X_BOUND = [-1, 2]  # x upper and lower bounds


# to find the maximum of this function
# def F(x): return np.sin(10 * x) * x + np.cos(2 * x) * x
# def F(x): return x * np.sin(10 * np.pi * x) + 2.0


# find non-zero fitness for selection
def get_fitness(pred):
    return pred


# convert binary DNA to decimal and normalize it to a rang(0,5)
def translateDNA(pop):
    # return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    select_value = pop.astype(np.bool)
    for i in select_value:
        data[i] = data[select_value]
    return data




# nature selection wrt pop fitness
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / fitness.sum())
    return pop[idx]


# roulette wheel selection
def select_gamble(pop, fitness):
    # sort by fitness
    sorted_index = np.argsort(fitness)
    sorted_pop = pop[sorted_index]  # 100,22
    sorted_fitness = fitness[sorted_index]  # 100,
    # out the time queue
    total_fitness = np.sum(sorted_fitness)

    accumulation = [None for col in range(len(sorted_fitness))]
    accumulation[0] = sorted_fitness[0] / total_fitness
    for i in range(1, len(sorted_fitness)):
        accumulation[i] = accumulation[i - 1] + sorted_fitness[i] / total_fitness
    accumulation = np.array(accumulation)

    # roulette wheel selection
    roulette_index = []
    for i in range(POP_SIZE):
        p = np.random.rand()
        for j in range(len(accumulation)):
            if float(accumulation[j]) >= p:
                roulette_index.append(j)
                break
    new_pop = []
    new_fitness = []
    for i in roulette_index:
        new_pop.append(sorted_pop[i])
        new_fitness.append(sorted_fitness[i])

    new_pop = np.array(new_pop)
    return new_pop


# mating process (genes crossover)
def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        # choose crossover points
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


# initialize the pop DNA
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))

for _ in range(N_GENERATIONS):
    # compute function value by extracting DNA
    #F_values = translateDNA(pop)
    data = data[:,pop[0]]
    accuracy_list = []
    accuracy_list.append(model.Neural_Network().__int__(data))

    # GA part(evolution)
    # fitness = get_fitness(F_values)
    fitness = np.array(fitness)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :], translateDNA(pop[np.argmax(fitness), :]))
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child

plt.ioff()
plt.show()
