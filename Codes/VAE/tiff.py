from PIL import Image
import os
import numpy as np
from random import sample
import nrrd
from scipy.ndimage import zoom

path = "./data/"
file_names = os.listdir(path)

# Randomly choose npy
random_npy = sample(file_names, 3)
for name in random_npy:
    print(name)
    file_name = os.path.join(path, name)
    arr = np.load(file_name)
    arr = np.squeeze(arr)
    #arr = zoom(arr[:, :, 9:209], 0.25)
    for i in range(arr.shape[0]):
        print(i)
        # arr[i] = arr[i].astype(np.int64)
        image = np.transpose(arr[i, :, :])
        print(image.shape)
        Image.fromarray(image).save("./image/{}_{}.tif".format(
            name[:-4],
            i
        ))
