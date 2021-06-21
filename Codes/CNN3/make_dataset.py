import csv
import numpy as np
import json
import pickle


def make_dataset(scRNA_file, all_gene, sequence_length=3000):
    """
    This function is used tp make dataset for training.
    :param scRNA_file: single cell RNA sequencing data.
    :param all_gene: all gene upstream sequence.
    :param sequence_length: length of sequence in X.
    :return: None.
    """

    cell_id = 0

    scRNA_data = {}
    with open(scRNA_file, 'r') as f:
        expression = csv.reader(f)
        i = 0
        for line in expression:
            if i == 0:
                i = 1
                continue
            gene_name = line[1]
            scRNA_data[gene_name] = line[2:]

    file = open(all_gene, 'r')
    all_gene = json.load(file)

    num = 0
    for key in scRNA_data:
        if key not in all_gene:
            continue
        num += 1

    one_hot = np.zeros([num, 4, sequence_length, 1])
    y = np.zeros([num], dtype=float)

    gene_list = []
    j = 0
    for key in scRNA_data:
        if key not in all_gene:
            continue
        gene_list.append(key)
        sequence = all_gene[key]
        i = 0
        for base in sequence:
            if base == 'A':
                one_hot[j, 0, i, 0] = 1
            if base == 'C':
                one_hot[j, 1, i, 0] = 1
            if base == 'G':
                one_hot[j, 2, i, 0] = 1
            if base == 'T':
                one_hot[j, 3, i, 0] = 1
            i += 1
        y[j] = scRNA_data[key][cell_id]
        j += 1

    file = open("./data/selected_gene.pkl", "wb")
    pickle.dump(gene_list, file)

    np.save(file="./data/X.npy", arr=one_hot)
    np.save(file="./data/Y.npy", arr=y)


if __name__ == '__main__':
    make_dataset("./data/MB_scRNA_seq.csv", "./data/all_gene_sequence.json")
