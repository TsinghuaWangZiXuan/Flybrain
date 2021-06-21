import json
import numpy as np


def make_one_hot(dna_path):
    """
    This function is used to represent DNA sequence with one_hot vectors.
    :param dna_path: DNA relative data path.
    :return: None.
    """

    # Load DNA sequence
    with open(dna_path + "positive.json", 'r') as pf:
        positive_dna = json.load(pf)
    with open(dna_path + "negative.json", 'r') as nf:
        negative_dna = json.load(nf)

    data_set = np.zeros([2 * len(positive_dna), 4, 3000, 1])
    label = np.zeros([2 * len(positive_dna)])

    # Convert sequence to vector
    i = 0
    for DNA in positive_dna:
        seq = positive_dna[DNA]
        j = 0
        label[i] = 1
        for base in seq:
            if base == 'A':
                data_set[i, 0, j, 0] = 1
            if base == 'C':
                data_set[i, 1, j, 0] = 1
            if base == 'G':
                data_set[i, 2, j, 0] = 1
            if base == 'T':
                data_set[i, 3, j, 0] = 1
            j += 1
        for k in range(3000 - len(seq)):
            # Using infinite number for padding, this site will be equal to zero after log and exp step in training
            data_set[i, 0, k + len(seq), 0] = 1e+6
            data_set[i, 1, k + len(seq), 0] = 1e+6
            data_set[i, 2, k + len(seq), 0] = 1e+6
            data_set[i, 3, k + len(seq), 0] = 1e+6
        i += 1

    i = 0
    for DNA in negative_dna:
        seq = negative_dna[DNA]
        j = 0
        for base in seq:
            if base == 'A':
                data_set[i + len(negative_dna), 0, j, 0] = 1
            if base == 'C':
                data_set[i + len(negative_dna), 1, j, 0] = 1
            if base == 'G':
                data_set[i + len(negative_dna), 2, j, 0] = 1
            if base == 'T':
                data_set[i + len(negative_dna), 3, j, 0] = 1
            j += 1
        for k in range(3000 - len(seq)):
            data_set[i + len(negative_dna), 0, k + len(seq), 0] = 1e+6
            data_set[i + len(negative_dna), 1, k + len(seq), 0] = 1e+6
            data_set[i + len(negative_dna), 2, k + len(seq), 0] = 1e+6
            data_set[i + len(negative_dna), 3, k + len(seq), 0] = 1e+6
        i += 1

    # Shuffle dataset and label
    state = np.random.get_state()
    np.random.shuffle(data_set)
    np.random.set_state(state)
    np.random.shuffle(label)

    np.save(arr=data_set, file="X.npy")
    np.save(arr=label, file="Y.npy")


if __name__ == '__main__':
    make_one_hot("./DNA/")
