import numpy as np
from matplotlib import pyplot as plt

norm = np.random.normal(loc=0, scale=1, size=(2000,))
bins = np.linspace(-3, 3, 50)
plt.hist(x=norm, bins=bins)
plt.show()
