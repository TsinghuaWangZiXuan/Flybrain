import csv
import numpy as np
import json

from tokenization import tokenization


def make_dataset(scRNA_file, all_gene, max_len):
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

    y = np.zeros([num * 2], dtype=float)

    sequence_dict = {}
    j = 0
    for key in all_gene:
        if key not in scRNA_data:
            continue
        y[j] = scRNA_data[key][cell_id]
        j += 1
        sequence_dict[key] = all_gene[key]

        y[j] = scRNA_data[key][cell_id]
        j += 1
        sequence_dict[key + '_reverse'] = all_gene[key + '_reverse']

    with open("./data/all_gene_sequence.txt", "w") as seq_file:
        for gene_name in sequence_dict:
            seq_file.write(sequence_dict[gene_name])
            seq_file.write('\n')

    sequence = tokenization('train', "./data/all_gene_sequence.txt")
    for i, s in enumerate(sequence):
        print(len(s))
        if len(s) > max_len:
            n = len(s) - max_len
            sequence[i] = s[n:]
        else:
            n_pad = max_len - len(s)
            s.extend([0] * n_pad)
            sequence[i] = s
    x = np.asarray(sequence)

    np.save(file="./data/X.npy", arr=x)
    np.save(file="./data/Y.npy", arr=y)


if __name__ == '__main__':
    make_dataset("./data/MB_scRNA_seq.csv", "./data/all_gene_sequence.json", 200)
