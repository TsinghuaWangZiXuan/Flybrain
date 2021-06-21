import h5py
import nrrd
import pickle
import time
import numpy as np
import multiprocessing

supervoxel = h5py.File('./data/SupervoxelClusteringData.mat', 'r')
superdata = supervoxel['labels'][:]


def scale(x, maximum, minimum):
    return (x - minimum) / (maximum - minimum)


def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))
    image = (image - mean) / np.sqrt(var)
    return image


def brain_activation(fn, th1, th2):
    """
    This function is used to map images to vectors.
    :param fn: image file name.
    :param th1: threshold 1.
    :param th2: threshold 2.
    :return: vector.
    """

    # Load and preprocess data
    nrrd_data, nrrd_head = nrrd.read(fn)
    labels = np.zeros(7066, dtype=float)
    nrrd_data = normalize(nrrd_data)
    minimum = 0
    for i in range(218):
        if np.max(nrrd_data[:, :, i] > 0):
            if minimum == 0:
                minimum = i
            if np.max(nrrd_data[:, :, i]) < np.max(nrrd_data[:, :, minimum]):
                minimum = i
    nrrd_data = scale(nrrd_data, np.max(nrrd_data), np.max(nrrd_data[:, :, minimum]))

    # Find voxels
    for i in range(superdata.shape[0]):
        image = superdata[i, :, :]  # (512,1024)
        nrrd_image = nrrd_data[:, :, i]  # superdata.shape[0]-i-1

        if np.max(nrrd_image) > 0:
            binary_matrix = np.rot90(nrrd_image, k=-1)  # (512,1024)
            x, y = np.where(binary_matrix >= th1)
            for indx, indy in zip(x, y):
                v = image[indx, indy]
                labels[int(v)] += binary_matrix[indx, indy]

    # Generate label vectors
    for i in range(len(labels)):
        if labels[i] > th2:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


def run_activation(nf):
    y = np.zeros([len(nf), 7066])
    num = 0
    temp = 0
    for name in nf:
        time_end = time.time()
        time_start = temp
        print("time cost", time_end - time_start, "s", ":", name, "------", num)
        temp = time_end
        file_name = "F:/images/" + name
        labels_n = brain_activation(file_name, 0.2, 100)
        y[num] = labels_n
        num = num + 1
    return y


if __name__ == '__main__':
    with open("./data/nrrd_filename.pkl", "rb") as nf:
        nrrd_filename = pickle.load(nf)

    cpu_number = 8

    file_number = len(nrrd_filename)
    data_size = file_number // cpu_number

    processes = []
    with multiprocessing.Pool(processes=cpu_number) as pool:
        for i in range(cpu_number):
            files = nrrd_filename[i * data_size:(i + 1) * data_size] if i < cpu_number-1 else nrrd_filename[i * data_size:]
            p = pool.apply_async(run_activation, (files,))
            processes.append(p)
        pool.close()
        pool.join()

    y = []
    for p in processes:
        y.append(p.get())
    y = np.concatenate(y, axis=0)

    np.save(file="./data/Y.npy", arr=y)
