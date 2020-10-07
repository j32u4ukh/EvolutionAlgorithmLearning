from evolution import Evolution
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return (np.sin(10 * x) + np.cos(2 * x)) * x


class FunctionOptimization(Evolution):
    def __init__(self, value_range, rna_size, n_population, mutation_rate):
        super().__init__(rna_size=rna_size, n_population=n_population, logger_name="FunctionOptimization")
        self.mutation_rate = mutation_rate
        self.value_range = value_range
        self.translation_dot = 2 ** np.arange(self.rna_size)[::-1]
        self.translation_multiplier = self.value_range[1] / float(2 ** self.rna_size - 1)

    def initPopulation(self):
        self.population = np.random.randint(2, size=(self.n_population, self.rna_size))

    def translation(self):
        return self.population.dot(self.translation_dot) * self.translation_multiplier

    def mutate(self, child):
        for r in range(self.rna_size):
            if np.random.rand() < self.mutation_rate:
                child[r] = 1 if child[r] == 0 else 0

        return child

    def evolve(self, *args, **kwargs):
        # 計算適應度並排序
        self.getFitness()

        # 繁殖
        self.reproduction()
        # self.logger.debug(f"n_population: {len(self.population)}")

        # 計算適應度並排序
        self.getFitness()

        # 淘汰機制
        self.naturalSelection()

    def naturalSelection(self, *args, **kwargs):
        """
        只保留較佳的基因組

        :param args:
        :param kwargs:
        :return:
        """
        self.population = self.population[-self.n_population:]
        # self.logger.debug(f"n_population: {len(self.population)}")

    def getFitness(self):
        x = self.translation()
        bound_limit = (self.value_range[0] <= x) & (x <= self.value_range[1])

        # 淘汰超出值域範圍的基因組
        self.population = self.population[bound_limit]

        # 計算適應度
        values = func(self.translation())

        fitness = np.argsort(values)

        highest_fitness = values[fitness[-1]]

        if highest_fitness > self.fitness:
            self.fitness = highest_fitness

            if self.potential < self.POTENTIAL:
                self.resetPotential()
        else:
            self.potential -= 1

        # 根據適應度排序
        self.population = self.population[fitness]

    def reproduction(self):
        # 取出最優秀的一批基因組
        n_best = int(self.N_POPULATION * self.reproduction_rate)
        # self.logger.debug(f"n_best: {n_best}")

        if n_best > self.n_population:
            n_best = self.n_population

        best = self.population[-n_best:]
        # self.logger.debug(f"#best: {len(best)}")

        for _ in range(n_best):
            # 取得兩個不重複的索引值
            idx = np.random.choice(np.arange(0, n_best), size=2, replace=False)
            # self.logger.debug(f"idx: {idx}")
            g1 = best[idx[0]]
            g2 = best[idx[1]]

            # 基因交換
            child = self.geneExchange(g1, g2)
            # self.logger.debug(f"基因交換 child: {child}")

            # 產生變異
            child = self.mutate(child)
            # self.logger.debug(f"產生變異 child: {child}")

            # 加入族群中
            # self.population.append(child)
            self.population = np.append(self.population, np.array([child]), axis=0)
            # self.logger.debug(f"n_population: {len(self.population)}")

    def geneExchange(self, *args):
        child = args[0].copy()
        other = args[1]

        # exchange_idx = [True,  True, False, ..., False, False]
        exchange_idx = np.random.randint(0, 2, size=self.rna_size).astype(np.bool)
        child[exchange_idx] = other[exchange_idx]

        return child


if __name__ == "__main__":
    RNA_SIZE = 10
    N_POPULATION = 100
    MUTATION_RATE = 0.003
    X_BOUND = [0, 5]
    N_GENERATIONS = 200

    fo = FunctionOptimization(value_range=X_BOUND,
                              rna_size=RNA_SIZE,
                              n_population=N_POPULATION,
                              mutation_rate=MUTATION_RATE)
    fo.naturalSelection()

    plt.ion()
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, func(x))

    for gen in range(N_GENERATIONS):
        if 'sca' in globals():
            sca.remove()

        x = fo.translation()
        y = func(x)
        # print(y)
        sca = plt.scatter(x,
                          y,
                          s=200,
                          lw=0,
                          c='red',
                          alpha=0.5)
        plt.pause(0.05)

        fo.evolve()
        print(func(fo.translation()[-1]))

        if fo.potential <= 0:
            break

    plt.ioff()
    plt.show()
