import numpy as np

TARGET_PHRASE = 'You get it!'       # target DNA
POP_SIZE = 300                      # population size
CROSS_RATE = 0.4                    # mating probability (DNA crossover)
# 變異機率
MUTATION_RATE = 0.01
N_GENERATIONS = 1000

DNA_SIZE = len(TARGET_PHRASE)
# 字串轉 ASCII
TARGET_ASCII = np.fromstring(TARGET_PHRASE, dtype=np.uint8)  # convert string to number
ASCII_BOUND = [32, 126]


class GA:
    def __init__(self, dna_size, dna_bound, cross_rate, mutation_rate, pop_size):
        self.DNA_size = dna_size
        dna_bound[1] += 1
        self.DNA_bound = dna_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.random.randint(*dna_bound, size=(pop_size, dna_size)).astype(np.int8)  # int8 for convert to ASCII

    @staticmethod
    # convert to readable string
    def translateDNA(dna):
        return dna.tostring().decode('ascii')

    # count how many character matches
    def get_fitness(self):
        match_count = (self.pop == TARGET_ASCII).sum(axis=1)
        return match_count

    def select(self):
        fitness = self.get_fitness() + 1e-4     # add a small amount to avoid all zero fitness
        idx = np.random.choice(np.arange(self.pop_size), 
                               size=self.pop_size, 
                               replace=True, 
                               p=fitness/fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            # select another individual from pop
            i_ = np.random.randint(0, self.pop_size, size=1)
            
            # choose crossover points
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)
            
            # mating and produce one child
            parent[cross_points] = pop[i_, cross_points]                            
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                # choose a random ASCII index
                child[point] = np.random.randint(*self.DNA_bound)  
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


if __name__ == '__main__':
    ga = GA(dna_size=DNA_SIZE, dna_bound=ASCII_BOUND, cross_rate=CROSS_RATE,
            mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)

    for generation in range(N_GENERATIONS):
        fitness = ga.get_fitness()
        best_DNA = ga.pop[np.argmax(fitness)]
        best_phrase = ga.translateDNA(best_DNA)
        print('Gen', generation, ': ', best_phrase)
        if best_phrase == TARGET_PHRASE:
            break
        ga.evolve()
