import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dna, seg):
        super().__init__()
        self.seq = []
        self.id = []
        self.length = 0

        # Segmentation
        for line in dna:
            if line[0] == '>':
                self.id.append(line[1:-1])
            else:
                l = len(line[:-1])
                if l <= seg:
                    self.length += 1
                    self.seq.append(line[:-1])
                else:
                    num = l // seg
                    rem = l % seg
                    for i in range(num):
                        self.length += 1
                        self.seq.append(line[i * seg:i * seg + seg])
                    if rem > 0:
                        self.length += 1
                        self.seq.append(line[-seg:])

        one_hot = np.zeros([self.length, 4, seg])

        # Make one-hot vector
        n = 0
        for seq in self.seq:
            j = seg-1
            for i in range(len(seq) - 1, -1, -1):
                if seq[i] == 'A':
                    one_hot[n, 0, j] = 1
                elif seq[i] == 'C':
                    one_hot[n, 1, j] = 1
                elif seq[i] == 'G':
                    one_hot[n, 2, j] = 1
                elif seq[i] == 'T':
                    one_hot[n, 3, j] = 1
                j -= 1
                if j < 0:
                    break
            n += 1
        self.one_hot = one_hot

    def __getitem__(self, index):
        return self.one_hot[index]

    def __len__(self):
        return self.length