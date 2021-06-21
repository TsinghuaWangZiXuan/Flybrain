import numpy as np
from scipy.ndimage import zoom
import os
import multiprocessing
import nrrd


def reduce(file_names, in_path, out_path, i):
    num = 0
    exist_files = os.listdir(out_path)
    for name in file_names:
        fname = name[:-5]+".npy"
        if fname in exist_files:
            continue
        file_name = in_path + name
        nrrd_data, nrrd_head = nrrd.read(file_name)
        nrrd_data = np.transpose(nrrd_data[:, :, 13:205], (2, 0, 1))
        nrrd_data = zoom(nrrd_data, 0.25, order=0)
        nrrd_data = np.expand_dims(nrrd_data, 0)
        np.save(arr=nrrd_data, file=out_path + name[:-5] + '.npy')
        num += 1
        print("process: {}; step {}; shape: {}".format(
            i,
            num,
            nrrd_data.shape
        ))


if __name__ == '__main__':
    in_path = 'G:/images/'
    out_path = './data/'
    file_names = os.listdir(in_path)
    file_number = len(file_names)
    data_size = file_number // 16

    processes = []
    for i in range(16):
        files = file_names[i * data_size:(i + 1) * data_size] if i < 15 else file_names[i * data_size:]
        p = multiprocessing.Process(target=reduce, args=(files, in_path, out_path, i, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
