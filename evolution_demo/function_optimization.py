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
        pass

    def getFitness(self):
        self.population = func(self.population)

    def reproduction(self):
        # 取出最優秀的一批基因組
        n_best = int(self.N_POPULATION * self.reproduction_rate)

        if n_best > self.n_population:
            n_best = self.n_population

        best = self.population[:n_best]

    def geneExchange(self, g1, g2):
        pass


if __name__ == "__main__":
    RNA_SIZE = 10
    N_POPULATION = 100
    MUTATION_RATE = 0.003
    X_BOUND = [0, 5]

    fo = FunctionOptimization(rna_size=RNA_SIZE, n_population=N_POPULATION, mutation_rate=MUTATION_RATE)
    fo.value_scale = X_BOUND[1]

