from abc import ABCMeta, abstractmethod
from utils import getLogger


class Evolution(metaclass=ABCMeta):
    def __init__(self, rna_size, n_population, logger_name="Evolution"):
        self.logger = getLogger(logger_name=logger_name)

        # rna 長度: 暫時不可變動
        self.rna_size = rna_size

        # 族群個數: 可變動
        self.n_population = n_population

        # 族群個數基數: 幾乎不變動
        self.N_POPULATION = n_population

        # 族群整體
        self.population = None

        # 初始化族群
        self.initPopulation()

        """
        族群的大小，除了會受到物理或生物因子的調控之外，生物本身的生物潛能（biotic potential）與
        環境的負荷量（carrying capacity），也會影響到族群的大小。
        對生物而言，族群的個體數至少要多少，才能維持族群不致於滅絕呢？
        """
        # 繁殖倍率，種族數量越少，倍率越高
        self.reproduction_rate = 0.2

    # 初始化族群
    @abstractmethod
    def initPopulation(self):
        pass

    # RNA 轉譯
    @abstractmethod
    def translation(self):
        pass

    # 基因變異
    @abstractmethod
    def mutate(self):
        pass

    # 繁殖
    @abstractmethod
    def reproduction(self):
        """
        定義哪些基因組來進行繁殖，以及每次繁殖多少子代。
        定義如何重組基因組來源們和自身的基因。

        :return: 子代
        """
        pass

    # 計算環境適應度
    @abstractmethod
    def getFitness(self):
        pass

    # 天擇
    @abstractmethod
    def naturalSelection(self, env):
        pass
