import numpy as np
from time import time

cost_time = 0
n_run = 100

for _ in range(n_run):
    x = np.random.rand(1000000, ) * 6 - 3
    start_time = time()
    less = x[x < -1.0]
    more = x[x > 1.0]
    equal = x[(-1.0 <= x) & (x <= 1.0)]
    # print("#less:", len(less))
    # print("#more:", len(more))
    # print("#equal:", len(equal))
    end_time = time()
    cost_time += end_time - start_time

# sol1: average cost_time: 0.026179437637329103
print("average cost_time:", cost_time / n_run)

arr = np.random.randint(low=0, high=10, size=10)
print(arr)
idx = np.argsort(arr)
print(idx)
arr_idx = arr[idx]
print("arr_idx:", arr_idx)
# indexs = np.abs(len(idx) - 1 - idx)
# print(indexs)
arr_indexs = arr_idx[::-1]
print("arr_indexs:", arr_indexs)