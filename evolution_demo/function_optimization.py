from evolution import Evolution
import numpy as np


def func(x):
    return (np.sin(10 * x) + np.cos(2 * x)) * x


class FunctionOptimization(Evolution):
    def __init__(self, rna_size, n_population, mutation_rate):
        super().__init__(rna_size=rna_size, n_population=n_population, logger_name="FunctionOptimization")
        self.mutation_rate = mutation_rate
        self.value_scale = None

    def initPopulation(self):
        self.population = np.random.randint(2, size=(self.n_population, self.rna_size))

    def translation(self):
        if self.value_scale is None:
            self.logger.error("self.value_scale is None")
            return

        return self.population.dot(2 ** np.arange(self.rna_size)[::-1]) / float(2**self.rna_size-1) * self.value_scale

    def mutate(self):
        for p in range(self.n_population):
            for r in range(self.rna_size):
                if np.random.rand() < self.mutation_rate:
                    self.population[p, r] = 1 if self.population[p, r] == 0 else 0

    def naturalSelection(self, env):
        # TODO: 計算適應度 -> 取出最優秀的一批基因組來繁殖 -> 計算適應度 -> 淘汰最差的一批基因組
        self.getFitness()
        self.reproduction()
        self.getFitness()

    def getFitness(self):
        self.population = func(self.population)

    def reproduction(self):
        # 取出最優秀的一批基因組
        n_best = int(self.N_POPULATION * self.reproduction_rate)

        if n_best > self.n_population:
            n_best = self.n_population

        best = self.population[:n_best]

        for _ in range(n_best):
            # 取得兩個不重複的索引值
            idx = np.random.choice(np.arange(0, n_best), size=2, replace=False)
            g1 = best[idx[0]]
            g2 = best[idx[1]]

            # 基因交換
            child = self.geneExchange(g1, g2)

            # 加入族群中
            self.population.append(child)

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

    fo = FunctionOptimization(rna_size=RNA_SIZE, n_population=N_POPULATION, mutation_rate=MUTATION_RATE)
    fo.value_scale = X_BOUND[1]

