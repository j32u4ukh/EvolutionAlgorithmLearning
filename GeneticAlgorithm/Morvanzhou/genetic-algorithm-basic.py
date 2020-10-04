import matplotlib.pyplot as plt
import numpy as np

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)

# 產生變異的機率
MUTATION_RATE = 0.003
N_GENERATIONS = 200
# x upper and lower bounds
X_BOUND = [0, 5]


# to find the maximum of this function
def F(x):
    return np.sin(10 * x) * x + np.cos(2 * x) * x


# 適應程度
def get_fitness(pred): 
    return pred + 1e-3 - np.min(pred)


# 解釋 DNA：DNA 與實際數值間的轉換
# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop):
    """
    利用 2 進制來取得數值，再除以最大值來得到 0 ~ 1 的數值，最後根據值域範圍縮放到 0 ~ 5
    :param pop:
    :return:
    """
    # np.arange(DNA_SIZE) = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # np.arange(DNA_SIZE)[::-1] = array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    # (100, 10) dot (10,) = (100,)
    # sum of 2 ** np.arange(DNA_SIZE)[::-1] = 1023 = float(2**DNA_SIZE-1)
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


# 適者生存，不適者淘汰
# nature selection wrt pop's fitness
def select(pop, fitness):
    """
    適應度越高，被抽中的機率越高，反之則越低。因此，重複抽到是有可能的。
    array([ 0,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  9, 10, 11, 12, 12, 13,
       14, 15, 15, 16, 17, 18, 19, 22, 24, 25, 27, 31, 36, 36, 36, 38, 39,
       41, 41, 42, 42, 43, 44, 46, 46, 47, 47, 48, 49, 51, 52, 52, 53, 54,
       54, 55, 55, 57, 58, 58, 61, 62, 62, 63, 64, 65, 69, 70, 70, 71, 71,
       74, 75, 75, 76, 77, 77, 77, 78, 78, 79, 79, 80, 80, 83, 86, 87, 88,
       89, 89, 89, 90, 92, 92, 93, 94, 94, 94, 95, 96, 97, 99, 99])

    :param pop: 族群
    :param fitness: 適應度
    :return:
    """
    # parameter p:選擇的機率，適應度高機率高，適應度低機率低
    # np.arange(POP_SIZE) = [0, 1, ..., 98, 99]
    # p: np.arange(POP_SIZE) 中每個元素被抽出的機率，因此長度與它等長
    idx = np.random.choice(np.arange(POP_SIZE), 
                           size=POP_SIZE, 
                           replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


# 父母基因的交叉配對
# mating process (genes crossover)
def crossover(parent, pop):     
    if np.random.rand() < CROSS_RATE:
        # 從母群挑一個配對對象
        i_ = np.random.randint(0, POP_SIZE, size=1)
        # cross_points = [True,  True, False, ..., False, False]
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        # True 的部分會被替換成對象的值，相當於父母雙方基因的結合
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
