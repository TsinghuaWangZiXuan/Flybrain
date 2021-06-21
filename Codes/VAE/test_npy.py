import numpy as np
import os

path = "./data/"
files = os.listdir(path)
num = 0
for f in files:
    name = path + f
    arr = np.load(name)
    num += 1
    print(num)
    print(arr.shape)