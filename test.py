import numpy as np
from time import time

mutate = 2.0
a = np.random.rand(100, 200)

for i in range(500):
    a *= np.random.rand(*a.shape) * mutate + 1e-5
    mutate *= 0.8

    if mutate < 1e-5:
        print(f"round {i} | mutate: {mutate}")
        break

    if len(a[a <= 0.0]) != 0:
        print(f"{i} | #(a <= 0.0): {len(a[a <= 0.0])}")
