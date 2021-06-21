import json
import numpy as np


def make_one_hot(janelia_file, vdrc_file, max_length=3000):
    """
    This function is used to represent DNA sequence with one_hot vectors.
    :param max_length: length of vector
    :param janelia_file: janelia dataset.
    :param vdrc_file: vdrc dataset.
    :return: None.
    """

    # Load DNA sequence
    jf = open(janelia_file, 'r')
    vf = open(vdrc_file, 'r')
    j_data = jf.readlines()
    v_data = vf.readlines()
    janelia_dna = {}
    vdrc_dna = {}
    dna_list = []

    # Convert vdrc to one hot
    i = 0
    num = 0
    for line in j_data:
        if line[0] == ">":
            num += 1
            temp = np.zeros([4, max_length], dtype=np.int)
            seq = j_data[i + 1][:-1]
            if len(seq) <= max_length:
                pos = 0
                for base in seq:
                    if base == "A":
                        temp[0, pos] = 1
                    if base == "C":
                        temp[1, pos] = 1
                    if base == "G":
                        temp[2, pos] = 1
                    if base == "T":
                        temp[3, pos] = 1
                    pos = pos + 1
                for k in range(max_length - len(seq)):
                    temp[0, pos] = 1e+6
                    temp[1, pos] = 1e+6
                    temp[2, pos] = 1e+6
                    temp[3, pos] = 1e+6
                    pos = pos + 1

            if len(seq) > max_length:
                for j in range(max_length):
                    if seq[j] == "A":
                        temp[0, j] = 1
                    if seq[j] == "C":
                        temp[1, j] = 1
                    if seq[j] == "G":
                        temp[2, j] = 1
                    if seq[j] == "T":
                        temp[3, j] = 1
            janelia_dna[line[1:-1]] = temp.tolist()
            dna_list.append(line[1:-1])
        i = i + 1

    # Convert vdrc to one hot
    i = 0
    for line in v_data:
        if line[0] == ">":
            num += 1
            temp = np.zeros([4, max_length], dtype=np.int)
            seq = v_data[i + 1][:-1]
            if len(seq) <= max_length:
                pos = 0
                for base in seq:
                    if base == "A":
                        temp[0][pos] = 1
                    if base == "C":
                        temp[1][pos] = 1
                    if base == "G":
                        temp[2][pos] = 1
                    if base == "T":
                        temp[3][pos] = 1
                    pos = pos + 1
                for k in range(max_length - len(seq)):
                    temp[0, pos] = 1e+6
                    temp[1, pos] = 1e+6
                    temp[2, pos] = 1e+6
                    temp[3, pos] = 1e+6
                    pos = pos + 1

            if len(seq) > max_length:
                for j in range(max_length):
                    if seq[j] == "A":
                        temp[0, j] = 1
                    if seq[j] == "C":
                        temp[1, j] = 1
                    if seq[j] == "G":
                        temp[2, j] = 1
                    if seq[j] == "T":
                        temp[3, j] = 1

            # Extend id to match standard
            id = line[1:-1]
            if len(id) < 8:
                j = len(id)
                for k in range(8 - j):
                    id = id[0:2] + "0" + id[2:len(id)]
            vdrc_dna[id] = temp.tolist()
            dna_list.append(id)
        i = i + 1

    print(num)
    # Dump data
    with open("./data/Janelia_DNA.json", "w") as jf:
        json.dump(janelia_dna, jf)
    with open("./data/VDRC_DNA.json", "w") as vf:
        json.dump(vdrc_dna, vf)
    with open("./data/DNA_list.json", 'w') as df:
        json.dump(dna_list, df)


if __name__ == '__main__':
    make_one_hot("./data/Janelia_DNA.txt", "./data/VDRC_DNA.txt")
