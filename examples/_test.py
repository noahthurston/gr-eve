import numpy as np
import random
from bitarray import bitarray
import matplotlib.pyplot as plt

outer = np.array([])

print(outer)

arr = [[1,2,3]]

for x in range(5):
	if len(outer) == 0:
		outer = np.array(arr)
	else:
		outer = np.append(outer, np.array(arr), axis=0)

print(outer)
