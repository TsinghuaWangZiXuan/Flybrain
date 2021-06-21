import time
import json
import h5py
import numpy as np


def supervoxel2dict(supervoxel_file):
    """
    This function is used to build dictionary for supervoxels.
    :param supervoxel_file: supervoxel data.
    :return: none
    """
    supervoxel = h5py.File(supervoxel_file)
    superdata = supervoxel['labels'][:]
    dic = {}
    for i in range(7066):
        a = [[], [], []]
        dic[i] = a
    temp = 0
    print(dic)
    for i in range(218):
        time_end = time.time()
        time_start = temp
        print("time cost", time_end - time_start, "s")
        temp = time_end
        image = superdata[i, :, :]
        image = np.rot90(image, k=1)
        for j in range(1024):
            for k in range(512):
                if int(image[j, k]) != 0:
                    dic[int(image[j, k])][0].append(j)
                    dic[int(image[j, k])][1].append(k)
                    dic[int(image[j, k])][2].append(i)

    with open("./data/supervoxel.json", 'w') as f:
        json.dump(dic, f)


if __name__ == '__main__':
    supervoxel2dict("./data/SupervoxelClusteringData.mat")
