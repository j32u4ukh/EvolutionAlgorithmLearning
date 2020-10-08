import numpy as np


ones = np.ones(shape=(3, 5))
twos = np.ones(shape=(3, 5)) * 2.0
nums = np.vstack((ones, twos))
print(nums)

ones = nums[:3]
twos = nums[-3:]
print(ones)
print(twos)

flop = np.random.randint(2, size=(3, 5)).astype(np.bool)
print(flop)

ones[flop] = twos[flop]
print(ones)
print(twos)
print(nums)
# import temp
#
# t = temp.Test()
# t.private()
# t.public()
