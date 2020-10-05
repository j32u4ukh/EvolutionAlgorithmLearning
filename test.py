import numpy as np


data_size = 3
value_scale = np.arange(data_size)[::-1]


def computeValue(one_hot: np.array):
    return one_hot.dot(2 ** value_scale)


array = np.random.randint(2, size=(5, data_size))

values = []
for arr in array:
    value = computeValue(arr)
    values.append(value)

idx = np.argsort(values)
print(idx)
print(values)

sort_array = array[idx]

values = []
for arr in sort_array:
    value = computeValue(arr)
    values.append(value)

idx = np.argsort(values)
print(idx)
print(values)

# import temp
#
# t = temp.Test()
# t.private()
# t.public()

