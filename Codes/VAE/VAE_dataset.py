import numpy as np
from torch.utils.data import Dataset
import os


class DataSet(Dataset):
    def __init__(self, file_path):
        super().__init__()

        # Path of data
        self.path = file_path
        self.file_names = os.listdir(file_path)

    def __getitem__(self, index):
        file_name = self.path + self.file_names[index]

        # Load data
        arr = np.load(file_name)
        # arr = np.transpose(arr, (2, 0, 1))
        # arr = zoom(arr[:, :, 9:209], 0.25)
        #arr = arr[:, :, 1:-1]
        # arr = np.array(arr, dtype='float')
        arr = np.array(arr, dtype=np.float)
        arr /= 255
        return arr

    def __len__(self):
        return len(self.file_names)
