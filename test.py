import numpy as np
from matplotlib import pyplot as plt

arr = np.random.rand(10, 5)
print("#arr:", len(arr))
values = arr.sum(axis=1)
over_two = np.where(values >= 2.0)
arr = arr[over_two]
print("#arr:", len(arr))
