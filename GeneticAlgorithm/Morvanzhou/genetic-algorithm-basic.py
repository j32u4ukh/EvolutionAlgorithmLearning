# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:12:19 2019

@author: j32u4ukh
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)

# 產生變異的機率
MUTATION_RATE = 0.003
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x upper and lower bounds

# to find the maximum of this function
def F(x):
    return np.sin(10 * x) * x + np.cos(2 * x) * x     


# 適應程度
def get_fitness(pred): 
    return pred + 1e-3 - np.min(pred)


# DNA 與實際數值間的轉換
# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop):    
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


# 適者生存，不適者淘汰
# nature selection wrt pop's fitness
def select(pop, fitness):
    # parameter p:選擇的機率，適應度高機率高，適應度低機率低
    idx = np.random.choice(np.arange(POP_SIZE), 
                           size=POP_SIZE, 
                           replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


# 父母基因的交叉配對
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

# 基因變異(隨機挑選基因產生變異)
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


# initialize the pop DNA
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   

# something about plotting
plt.ion()
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))    # compute function value by extracting DNA

    # something about plotting
    if 'sca' in globals(): 
        sca.remove()
    sca = plt.scatter(translateDNA(pop), 
                      F_values, 
                      s=200, 
                      lw=0, 
                      c='red', 
                      alpha=0.5)
    plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

plt.ioff()
plt.show()
