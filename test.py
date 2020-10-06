import numpy as np


data_size = 3
value_scale = np.arange(data_size)[::-1]


def computeValue(one_hot: np.array):
    return one_hot.dot(2 ** value_scale)


array = np.random.randint(2, size=(5, data_size))
print(array)
values = computeValue(array)
print(values)
bound_limit = (0 <= values) & (values <= 5)
print(bound_limit)
array = array[bound_limit]
print(array)
values = computeValue(array)
print(values)
idx = np.argsort(values)
print(idx)
array = array[idx]
values = computeValue(array)
print(values)
idx = np.argsort(values)
print(idx)


# import temp
#
# t = temp.Test()
# t.private()
# t.public()
