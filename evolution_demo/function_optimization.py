from evolution import Evolution
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return (np.sin(10 * x) + np.cos(2 * x)) * x


class SimpleExchangeDemo(Evolution):
    def __init__(self, value_range, rna_size, n_population, mutation_rate):
        super().__init__(rna_size=rna_size, n_population=n_population, logger_name="SimpleExchangeDemo")
        self.mutation_rate = mutation_rate
        self.value_range = value_range
        # np.arange(self.rna_size): 產生 [0, 1, ..., self.rna_size - 1]
        self.translation_dot = 2 ** np.arange(self.rna_size)[::-1]
        self.translation_multiplier = self.value_range[1] / float(2 ** self.rna_size - 1)

    def initPopulation(self):
        # self.population.shape = (self.n_population, self.rna_size)
        # self.population = [[0, 1, 1, 0, ...], ..., [1, 1, 0, 0, ...]]
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

        # 根據 values 數值大小，給予排名的數列，數值越小，排名數值越小
        # np.argsort([9, 4, 6]) -> array([1, 2, 0], dtype=int64)
        fitness = np.argsort(values)

        average_fitness = values.mean()

        if average_fitness > self.fitness:
            self.fitness = average_fitness

            if self.potential < self.POTENTIAL:
                self.resetPotential()
        else:
            self.potential -= 1

        # 根據適應度排序: 讓最小的排最前面，最大的排最後面
        self.population = self.population[fitness]

    def reproduction(self):
        # 取出最優秀的一批基因組
        n_best = int(self.N_POPULATION * self.reproduction_rate)

        if n_best > self.n_population:
            n_best = self.n_population

        best = self.population[-n_best:]

        for _ in range(n_best):
            # np.random.choice -> replace=False: 取得範圍內的兩個值(不重複)；replace=True: 取得範圍內的兩個值(可能重複)
            idx = np.random.choice(np.arange(0, n_best), size=2, replace=False)

            g1 = best[idx[0]]
            g2 = best[idx[1]]

            # 基因交換
            child = self.geneExchange(g1, g2)

            # 產生變異
            child = self.mutate(child=child)

            # 加入族群中
            # np.append 返回添加後的結果，不改變原始陣列，將陣列 2 加到陣列 1 當中，沿著指定的 axis
            self.population = np.append(self.population, np.array([child]), axis=0)

    def geneExchange(self, *args):
        child = args[0].copy()
        other = args[1]

        # exchange_idx = [True,  True, False, ..., False, False]
        exchange_idx = np.random.randint(0, 2, size=self.rna_size).astype(np.bool)
        child[exchange_idx] = other[exchange_idx]

        return child


class WinnerLoserDemo(Evolution):
    def __init__(self, value_range, rna_size, n_population, mutation_rate):
        super().__init__(rna_size=rna_size, n_population=n_population, logger_name="WinnerLoserDemo")
        self.mutation_rate = mutation_rate
        self.value_range = value_range
        # np.arange(self.rna_size): 產生 [0, 1, ..., self.rna_size - 1]
        self.translation_dot = 2 ** np.arange(self.rna_size)[::-1]
        self.translation_multiplier = self.value_range[1] / float(2 ** self.rna_size - 1)

    def initPopulation(self):
        self.population = np.random.randint(2, size=(self.n_population, self.rna_size))

    def translation(self):
        return self.population.dot(self.translation_dot) * self.translation_multiplier

    def mutate(self, child: np.array):
        flop = np.random.randint(2, size=child.shape).astype(np.bool)

    def reproduction(self, *args, **kwargs):
        pass

    def geneExchange(self, *args, **kwargs):
        pass

    def getFitness(self, *args, **kwargs):
        pass

    def naturalSelection(self, *args, **kwargs):
        pass

    def evolve(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    def testSimpleExchangeDemo():
        """
        將畫圖相關程式碼移到函式中，會導致互動效果出錯，但並非程式碼本身出錯。

        :return:
        """
        RNA_SIZE = 10
        N_POPULATION = 100
        MUTATION_RATE = 0.003
        X_BOUND = [0, 5]
        N_GENERATIONS = 200

        sed = SimpleExchangeDemo(value_range=X_BOUND,
                                 rna_size=RNA_SIZE,
                                 n_population=N_POPULATION,
                                 mutation_rate=MUTATION_RATE)
        sed.naturalSelection()

        plt.ion()
        x = np.linspace(*X_BOUND, 200)
        plt.plot(x, func(x))

        for gen in range(N_GENERATIONS):
            if 'sca' in globals():
                sca.remove()

            x = sed.translation()
            y = func(x)
            # print(y)
            sca = plt.scatter(x,
                              y,
                              s=200,
                              lw=0,
                              c='red',
                              alpha=0.5)
            plt.pause(0.05)

            sed.evolve()
            print(gen, func(sed.translation()[-1]))

            if sed.potential <= 0:
                break

        plt.ioff()
        plt.show()


    testSimpleExchangeDemo()
