import matplotlib.pyplot as plt
import numpy as np

# DNA size
N_CITIES = 20
CROSS_RATE = 0.1
# 變異機率
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 500


class GA:
    def __init__(self, dna_size, cross_rate, mutation_rate, pop_size):
        self.DNA_size = dna_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        # np.random.permutation(dna_size): 長度為 dna_size 的數列並打亂
        self.pop = np.vstack([np.random.permutation(dna_size) for _ in range(pop_size)])

    @staticmethod
    def translateDNA(dna, city_position):     # get cities' coord in order
        line_x = np.empty_like(dna, dtype=np.float64)
        line_y = np.empty_like(dna, dtype=np.float64)
        for i, d in enumerate(dna):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), 
                               size=self.pop_size, 
                               replace=True, 
                               p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            # select another individual from pop
            i_ = np.random.randint(0, self.pop_size, size=1)

            # choose crossover points: array([0, 1, 1, ..., 1, 0]) -> array([ False, True,  True, ..., True, False])
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)

            # find the city number
            keep_city = parent[~cross_points]

            # np.ravel() 和 np.flatten() 都是將多維陣列降為一維陣列，兩者的區別在於
            # np.flatten() 返回一份拷貝，對拷貝所做的修改不會影響（reflects）原始矩陣，
            # np.ravel()返回的是視圖（view），會影響（reflects）原始矩陣。
            # np.isin(a, b) 用於判定 a 中的元素在 b 中是否出現過，如果出現過返回 True,否則返回 False,
            # 最終結果為一個形狀和 a 一模一樣的陣列。
            # 但是當引數 invert 被設定為 True 時，情況恰好相反，如果 a 中元素在 b 中沒有出現則返回 True,如果出現了則返回 False.
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swap_a, swap_b = child[point], child[swap_point]
                child[point], child[swap_point] = swap_b, swap_a
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


class TravelSalesPerson:
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)


ga = GA(dna_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

env = TravelSalesPerson(N_CITIES)
for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)
    fitness, total_distance = ga.get_fitness(lx, ly)
    ga.evolve(fitness)
    best_idx = np.argmax(fitness)
    print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)

    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

plt.ioff()
plt.show()
