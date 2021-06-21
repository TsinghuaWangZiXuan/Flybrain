import numpy as np
import scipy.io as io

numpy_file1 = np.load('test.npy')
io.savemat('test.mat', {'test': numpy_file1})
