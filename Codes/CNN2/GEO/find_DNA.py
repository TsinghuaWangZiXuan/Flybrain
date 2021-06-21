import csv
import os
import json


def find_dna(dna_path):
    """
    This function is mainly used to get DNA sequence of fly lines which show GFP expression in DC3 adPN.
    Data come from NeuronBridge.
    :param dna_path: DNA relative data path.
    :return: None.
    """

    # Scree lines that possess expression
    dna_id_list = []
    dna_file = os.listdir(dna_path)
    dna_list = []
    for file in dna_file:
        filename = dna_path + file
        if filename[-3:] != 'csv':
            continue
        with open(filename, 'r') as c:
            dna = csv.reader(c)
            i = 0
            for line in dna:
                if i == 0:
                    i = 1
                    continue
                if float(line[2]) > 10000:  # Expression score must exceed 10000
                    id = line[1]
                    if id[0] == 'R':
                        id = "GM" + id
                    if id not in dna_id_list:
                        dna_id_list.append(id)
                if id not in dna_list:
                    dna_list.append(id)

    dna = {}
    # Get DNA sequence from Janelia dataset
    with open(dna_path + "Janelia_DNA.txt") as jf:
        jf = jf.readlines()
        i = 0
        for line in jf:
            if line[0] == '>':
                id = line[1:-1]
                dna[id] = jf[i + 1]
            i += 1

    # Get DNA sequence from VDRC dataset
    with open(dna_path + "VDRC_DNA_new.txt") as vf:
        vf = vf.readlines()
        i = 0
        for line in vf:
            if line[0] == '>':
                id = line[1:-1]
                dna[id] = vf[i + 1]
            i += 1

    # Choose positive data
    num = 0
    positive_dna = {}
    for line in dna_id_list:
        if line not in dna:
            print(line)
        else:
            if 3000 >= len(dna[line][:-1]) >= 1000:  # DNA sequence length [1000:3000]
                positive_dna[line] = dna[line][:-1]
                num += 1

    # Save positive DNA
    with open("./DNA/positive.json", 'w') as f:
        json.dump(positive_dna, f)

    # Choose negative data
    negative_dna = {}
    for i in range(int(num / 2)):
        with open("./DNA/Janelia_DNA.txt") as jf:
            jf = jf.readlines()
            j = 0
            for line in jf:
                if line[0] == '>':
                    id = line[1:-1]
                    seq = jf[j + 1][:-1]
                    if id not in dna_list and id not in negative_dna and 2000 < len(seq) < 3000:
                        negative_dna[id] = seq
                        break
                j += 1
        with open("./DNA/VDRC_DNA_new.txt") as vf:
            vf = vf.readlines()
            j = 0
            for line in vf:
                if line[0] == '>':
                    id = line[1:-1]
                    seq = vf[j + 1][:-1]
                    if id not in dna_list and id not in negative_dna and 2000 < len(seq) < 3000:
                        negative_dna[id] = seq
                        break
                j += 1

    # Save negative DNA
    with open(dna_path + "negative.json", 'w') as f:
        json.dump(negative_dna, f)
    print(len(positive_dna))
    print(len(negative_dna))


if __name__ == '__main__':
    find_dna("./DNA/")
