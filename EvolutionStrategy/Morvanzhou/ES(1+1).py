"""
(1+1)-ES with 1/5th success rule with visualization.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1             # DNA (real number)
DNA_BOUND = [0, 5]       # solution upper and lower bounds
N_GENERATIONS = 200
MUT_STRENGTH = 5.        # initial step size (dynamic mutation strength)


def F(x):
    # to find the maximum of this function
    return np.sin(10*x)*x + np.cos(2*x)*x


# find non-zero fitness for selection
def get_fitness(pred):
    return pred.flatten()


def make_kid(parent):
    # 變異
    # no crossover, only mutation
    k = parent + MUT_STRENGTH * np.random.randn(DNA_SIZE)

    # 確保位於值域範圍中
    k = np.clip(k, *DNA_BOUND)
    return k


def kill_bad(parent, kid):
    global MUT_STRENGTH
    fp = get_fitness(F(parent))[0]
    fk = get_fitness(F(kid))[0]
    p_target = 1/5

    # 子代比親代優秀 -> MUT_STRENGTH *= 大 -> 持續變異，探索可能空間 2.028114981647472
    if fp < fk:     # kid better than parent
        parent = kid
        ps = 1.     # kid win -> ps = 1 (successful offspring)

    # 親代比子代優秀 -> MUT_STRENGTH *= 小 -> 變異收斂 0.8379668855787558
    else:
        ps = 0.

    # 2.028114981647472 / 0.8379668855787558 = 2.4202805821458218
    # adjust global mutation strength
    MUT_STRENGTH *= np.exp(1/np.sqrt(DNA_SIZE+1) * (ps - p_target)/(1 - p_target))
    return parent


parent = 5 * np.random.rand(DNA_SIZE)   # parent DNA

plt.ion()
x = np.linspace(*DNA_BOUND, 200)

for _ in range(N_GENERATIONS):
    # ES part
    kid = make_kid(parent)
    py, ky = F(parent), F(kid)       # for later plot
    parent = kill_bad(parent, kid)

    # something about plotting
    plt.cla()
    plt.scatter(parent, py, s=200, lw=0, c='red', alpha=0.5,)
    plt.scatter(kid, ky, s=200, lw=0, c='blue', alpha=0.5)
    plt.text(0, -7, 'Mutation strength=%.2f' % MUT_STRENGTH)
    plt.plot(x, F(x))
    plt.pause(0.05)

plt.ioff()
plt.show()
