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

        # TODO: 若適應度無法再提升，應提前終止演化
        self.fitness = 0
        self.POTENTIAL = 5
        self.potential = 5

    # 初始化族群
    @abstractmethod
    def initPopulation(self):
        pass

    def setPotential(self, potential=5):
        """
        適應度無法再提升的情況下，應再嘗試演化幾輪的定義。

        :param potential: 演化幾輪
        :return:
        """
        self.POTENTIAL = potential
        self.potential = potential

    def resetPotential(self):
        """
        之前在適應度無法再提升的情況下，隨著嘗試演化數輪，適應度再次提升，因此須將嘗試次數還原，
        以便下次出現適應度無法再提升的情況時，可以再次使用。

        :return:
        """
        self.potential = self.POTENTIAL

    # RNA 轉譯
    @abstractmethod
    def translation(self):
        pass

    # 基因變異
    @abstractmethod
    def mutate(self, *args, **kwargs):
        pass

    # 繁殖
    @abstractmethod
    def reproduction(self, *args, **kwargs):
        """
        定義哪些基因組來進行繁殖，以及每次繁殖多少子代。
        定義如何重組基因組來源們和自身的基因。

        :return: 子代
        """
        pass

    # 基因交換
    @abstractmethod
    def geneExchange(self, *args, **kwargs):
        pass

    # 計算環境適應度
    @abstractmethod
    def getFitness(self, *args, **kwargs):
        pass

    # 天擇
    @abstractmethod
    def naturalSelection(self, *args, **kwargs):
        """
        定義淘汰機制。

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def evolve(self, *args, **kwargs):
        """
        封裝 '''計算適應度 -> 繁殖 -> 計算適應度 -> 淘汰''' 的過程
        :param args:
        :param kwargs:
        :return:
        """
        pass
