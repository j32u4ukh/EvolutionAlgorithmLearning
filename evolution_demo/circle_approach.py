import numpy as np
from matplotlib import pyplot as plt
from evolution import Evolution


def circle(x=None, y=None):
    CENTER_X = 0.0
    CENTER_Y = 0.0
    RADIUS = 3.0

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
    def __init__(self):
        pass

    def initPopulation(self):
        pass

    def translation(self):
        pass

    def evolve(self, *args, **kwargs):
        pass

    def getFitness(self, *args, **kwargs):
        pass

    def reproduction(self, *args, **kwargs):
        pass

    def geneExchange(self, *args, **kwargs):
        pass

    def mutate(self, *args, **kwargs):
        pass

    def naturalSelection(self, *args, **kwargs):
        pass


RADIUS = 3.0
N_GENERATIONS = 200

ca = CircleApproach()
x = np.hstack((np.linspace(-RADIUS, -RADIUS + 0.1, num=100),
               np.linspace(-RADIUS, RADIUS, num=500),
               np.linspace(RADIUS - 0.1, RADIUS, num=100)))
x, y = circle(x=x)

# plt.ion()
plt.figure(figsize=(RADIUS * 2.0, RADIUS * 2.0))
plt.xlim(-RADIUS - 2.0, RADIUS + 2.0)
plt.ylim(-RADIUS - 2.0, RADIUS + 2.0)
plt.scatter(x, y)

# for gen in range(N_GENERATIONS):
#     ca.evolve()
#     scatter = plt.scatter(x, y)
#     plt.pause(0.05)
#     scatter.remove()
#
# plt.ioff()
# plt.show()
