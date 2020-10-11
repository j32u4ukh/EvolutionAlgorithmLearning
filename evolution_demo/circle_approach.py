import numpy as np
from matplotlib import pyplot as plt
from evolution import Evolution

CENTER_X = 0.0
CENTER_Y = 0.0
RADIUS = 3.0
CENTER = [CENTER_X, CENTER_Y]

def circle(x=None, y=None):
    if x is not None:
        data = x
        center = CENTER_X
    else:
        data = y
        center = CENTER_Y

    data = data[(center - RADIUS <= data) & (data <= center + RADIUS)]
    output = np.ones_like(data) * RADIUS ** 2.0 - np.power(data, 2.0)
    square_root_output = np.power(output, 0.5)
    pos_output = center + square_root_output
    neg_output = center - square_root_output

    return np.hstack((data, data)), np.hstack((pos_output, neg_output))


class CircleApproach(Evolution):
    def __init__(self, value_range, fragment_size, n_population, exchange_rate, center, radius):
        self.value_range = value_range
        self.exchange_rate = exchange_rate
        self.mutation_strength = 2.0
        super().__init__(fragment_size=fragment_size, n_fragment=2,
                         n_population=n_population, logger_name="CircleApproach")

        self.center = center
        self.radius = radius

    def initPopulation(self):
        """
        [[mu_x, mu_y, std_x, std_y],
        ...,
        [mu_x, mu_y, std_x, std_y]]
        :return:
        """
        value_range = self.value_range[1] - self.value_range[0]

        # 同時建立 X 和 Y 的基因組，但值域範圍處理時要用單一基因組個數來計算
        mu = (np.random.rand(self.n_population, self.rna_size) - 0.5) * value_range / self.fragment_size
        std = np.random.rand(self.n_population, self.rna_size) * self.mutation_strength + 1e-5
        self.population = np.hstack((mu, std))

    def translation(self):
        mu_x = self.getMuX()
        mu_y = self.getMuY()

        return mu_x.sum(axis=1), mu_y.sum(axis=1)

    def evolve(self, *args, **kwargs):
        # 計算適應度並排序
        distance, _ = self.getFitness()

        # 繁殖
        self.reproduction(distance=distance)

        # 計算適應度並排序
        _, fitness = self.getFitness()

        # Early stop
        self.modifyEarlyStop(fitness=fitness)

        # 淘汰機制
        self.naturalSelection(fitness=fitness)

    def getFitness(self, *args, **kwargs):
        # x, y = self.translation()
        # valid_x = np.where((self.value_range[0] <= x) & (x <= self.value_range[1]))
        # valid_y = np.where((self.value_range[0] <= y) & (y <= self.value_range[1]))
        # valid_idx = np.intersect1d(valid_x, valid_y)
        #
        # # 淘汰超出值域範圍的基因組
        # self.population = self.population[valid_idx]
        # self.updatePopulationSize(n_population=len(self.population))
        # self.logger.info(f"淘汰超出值域範圍的基因組 #population: {self.n_population}")

        x, y = self.translation()
        square = np.square(x - self.center[0]) + np.square(y - self.center[1])
        root = np.sqrt(square)

        # 距離圓周的距離，大於 0 表示在圓之外，小於 0 表示在圓之內
        distance = self.radius - root

        # 適應度: 距離圓周越近越好 <= self.radius
        fitness = self.radius - np.abs(distance)

        # 根據 values 數值大小，給予排名的數列，數值越小，排名數值越小
        # np.argsort([9, 4, 6]) -> array([1, 2, 0], dtype=int64)
        indexs = np.argsort(fitness)

        # 根據適應度排序: 讓最小的排最前面，最大的排最後面
        self.population = self.population[indexs]

        # 距離圓周的距離也給予排序
        distance = distance[indexs]

        return distance, fitness

    def reproduction(self, *args, **kwargs):
        distance = kwargs['distance']
        self.logger.info(f"distance: {distance}")
        inside = self.population[np.where(distance < self.radius - 1e-5)]
        outside = self.population[np.where(distance > self.radius + 1e-5)]
        # online = self.population[np.where(self.radius - 1e-5 <= distance <= self.radius + 1e-5)]

        n_reproduction = min(len(inside), len(outside))

        # TODO: 考慮只有 inside 或只有 outside 的情況!!
        # getFitness 當中根據適應度排序，絕對距離越小的會在越後面，即便現在根據實際差距取出，此排序還是存在
        inside = inside[-n_reproduction:]
        outside = outside[-n_reproduction:]

        for _ in range(self.reproduction_scale):
            # 基因交換
            children = self.geneExchange(inside=inside, outside=outside)

            children = self.mutate(children=children)

            # 加入族群中
            self.addOffspring(offspring=children)
            self.updatePopulationSize(n_population=len(self.population))
            self.logger.info(f"#population: {self.n_population}")

    def geneExchange(self, *args, **kwargs):
        # shape = (n_reproduction, fragment_size * 4)
        children = kwargs["inside"].copy()
        n_children, _ = children.shape

        # mu 和相對應的 std 一起進行交換
        flop_x = np.random.rand(n_children, self.fragment_size) < self.exchange_rate
        flop_y = np.random.rand(n_children, self.fragment_size) < self.exchange_rate
        flop = np.hstack((flop_x, flop_y, flop_x, flop_y))

        children[flop] = kwargs["outside"][flop]
        return children

    def mutate(self, *args, **kwargs):
        mu_x, std_x, mu_y, std_y = self.getGenome(population=kwargs["children"])

        # 以原始 mu_x 為平均數，標準差為 std_x 的常態分配重新抽樣
        mu_x = np.random.normal(loc=mu_x, scale=std_x)

        # 對 std_x 的變異強度做增減
        std_x *= (np.random.rand(*std_x.shape) * self.mutation_strength + 1e-5)

        # 以原始 mu_y 為平均數，標準差為 std_y 的常態分配重新抽樣
        mu_y = np.random.normal(loc=mu_y, scale=std_y)

        # 對 std_y 的變異強度做增減
        std_y *= (np.random.rand(*std_y.shape) * self.mutation_strength + 1e-5)

        return np.hstack((mu_x, std_x, mu_y, std_y))

    def modifyEarlyStop(self, *args, **kwargs):
        average_fitness = kwargs['fitness'].mean()

        if average_fitness > self.fitness:
            self.fitness = average_fitness

            self.mutation_strength /= 0.8
        else:
            if self.mutation_strength < 1e-5:
                self.potential = 0

            self.mutation_strength *= 0.8

    def naturalSelection(self, *args, **kwargs):
        # 只保留適應度大於 0 的基因組
        self.population = self.population[np.where(kwargs['fitness'] > 0.0)]
        self.population = self.population[-self.N_POPULATION]
        self.updatePopulationSize(n_population=len(self.population))

    def getGenome(self, population=None):
        if population is None:
            population = self.population

        mu_x = self.getMuX(population=population)
        std_x = self.getStdX(population=population)
        mu_y = self.getMuY(population=population)
        std_y = self.getStdY(population=population)

        return mu_x, std_x, mu_y, std_y

    def getMuX(self, population=None):
        if population is None:
            population = self.population

        return population[:, :self.fragment_size]

    def getMuY(self, population=None):
        if population is None:
            population = self.population

        return population[:, self.fragment_size:self.fragment_size * 2]

    def getStdX(self, population=None):
        if population is None:
            population = self.population

        return population[:, self.rna_size:self.rna_size + self.fragment_size]

    def getStdY(self, population=None):
        if population is None:
            population = self.population

        return population[:, self.rna_size + self.fragment_size:]


FRAGMENT_SIZE = 10
N_POPULATION = 200
EXCHANGE_RATE = 0.6
MUTATION_RATE = 0.003
VALUE_RANGE = [-RADIUS, RADIUS]
N_GENERATIONS = 200

ca = CircleApproach(value_range=VALUE_RANGE,
                    fragment_size=FRAGMENT_SIZE,
                    n_population=N_POPULATION,
                    exchange_rate=EXCHANGE_RATE,
                    center=CENTER,
                    radius=RADIUS)
x = np.hstack((np.linspace(-RADIUS, -RADIUS + 0.1, num=100),
               np.linspace(-RADIUS, RADIUS, num=500),
               np.linspace(RADIUS - 0.1, RADIUS, num=100)))
x, y = circle(x=x)

plt.ion()
plt.figure(figsize=(RADIUS * 2.0, RADIUS * 2.0))
plt.xlim(-RADIUS - 2.0, RADIUS + 2.0)
plt.ylim(-RADIUS - 2.0, RADIUS + 2.0)
plt.scatter(x, y)

for gen in range(N_GENERATIONS):
    ca.evolve()
    x, y = ca.translation()
    scatter = plt.scatter(x, y)
    plt.pause(0.05)
    scatter.remove()

plt.ioff()
plt.show()
