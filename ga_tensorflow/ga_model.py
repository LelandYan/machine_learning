# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/17 23:02'

import numpy as np
import pandas as pd
import model

# the path and name of file
CSV_FILE_PATH = 'csv_result-colonTumor.csv'
# read the file
df = pd.read_csv(CSV_FILE_PATH)
shapes = df.values.shape
# the eigenvalue of file
input_data = df.values[:, 1:shapes[1] - 1]
# the result of file
result = df.values[:, shapes[1] - 1:shapes[1]]
# the length of eigenvalue
value_len = input_data.shape[1]
# the length of result
pop_len = result.shape[0]
# DNA length
DNA_SIZE = value_len
# population size
POP_SIZE = pop_len
# mating probability(DNA crossover)
CROSS_RATE = 0.8
# mutation probability
MUTATION_RATE = 0.003
# the times of generations
N_GENERATIONS = 100


# find non-zero fitness for selection
def get_fitness(pred):
    return pred


# convert binary DNA to decimal and normalize it to a rang(0,5)
def translateDNA(pop):
    index_list = []
    for i in range(len(pop)):
        if pop[i] == 1:
            index_list.append(i)
    return index_list


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


# genovariation
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


# pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))

# initialize the pop DNA
pop = np.zeros((POP_SIZE, DNA_SIZE))
pop = np.full(pop.shape, 0)
count = 1

# pick up the 20 points from DNA
for i in range(len(pop)):
    for j in range(len(pop[i])):
        if count <= 0.005 * DNA_SIZE:
            if np.random.rand() < 0.8:
                pop[i][j] = 1
                count += 1

# the training of ga
for _ in range(N_GENERATIONS):
    accuracy_list = []
    feature_list = []
    for i in range(input_data.shape[0]):
        data = input_data[:, translateDNA(pop[i])]
        # data = data[:, pop[i]]
        feature_list.append(np.sum(pop, axis=1)[0])
        accuracy_list.append(model.Neural_Network().__int__(data, result)[0])
    # GA part(evolution)
    fitness = np.array(accuracy_list)
    features = np.array(feature_list)
    print("accuracy: ", np.max(accuracy_list), " features: ", features[np.argmax(accuracy_list)])
    pop = select_gamble(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child
