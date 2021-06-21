import numpy as np

a = [0]

b = np.log(a)

print(b)

b[np.isinf(b)] = -1e+6

print(b)
