import numpy as np
import matplotlib.pyplot as plt

# DNA length
DNA_SIZE = 10
# population size
POP_SIZE = 20
# mating probability(DNA crossover)
CROSS_RATE = 0.6
# mutation probability
MUTATION_RATE = 0.01
N_GENERATOINS = 200
# x upper and lower bounds
X_BOUND = [0, 5]


# to find the maximum of this function
def F(x): return np.sin(10 * x) + np.cos(2 * x) * x


class MGA(object):
    def __int__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        # initial DNAs for winner and loser
        self.pop = np.random.randint(*DNA_bound, size=(1, self.DNA_size)).repeat(pop_size, axis=0)
