import csv
import os
import json
import numpy as np


def get_sequence(gene_loci, chr_dict, up_length=1000):
    """
    This function is used for get upstream sequence for given gene location and chromosome sequence.
    :param up_length:  the length of upstream sequence. Default=3000.
    :param gene_loci: gene location.
    :param chr_dict: all chromosome sequence.
    :return: up stream sequence
    """

    # Find gene location
    position = 1 if gene_loci[-3:] == '(1)' else -1
    if position == 1:
        s = 0
        loci = ''
        for i in gene_loci:
            if i == ':':
                s = 1
                continue
            if i == '.':
                break
            if s == 1:
                loci += i
        start = int(loci) - up_length
        end = int(loci)
    else:
        s = 0
        loci = ''
        for i in gene_loci:
            if i == '.':
                s = 1
                continue
            if i == '(':
                break
            if s == 1 and i != '.':
                loci += i
        start = int(loci)
        end = int(loci) + up_length

    # Get sequence
    if gene_loci[:2] == '2L':
        sequence = chr_dict['chr2L'][start:end]
    elif gene_loci[:2] == '2R':
        sequence = chr_dict['chr2R'][start:end]
    elif gene_loci[:2] == '3L':
        sequence = chr_dict['chr3L'][start:end]
    elif gene_loci[:2] == '3R':
        sequence = chr_dict['chr3R'][start:end]
    elif gene_loci[0] == '4':
        sequence = chr_dict['chr4'][start:end]
    elif gene_loci[0] == 'X':
        sequence = chr_dict['chrX'][start:end]
    elif gene_loci[0] == 'Y':
        sequence = chr_dict['chrY'][start:end]
    else:
        sequence = ""

    return sequence


def get_upstream_sequence(rg_path, gene_loci_file, up_length=3000):
    """
    This function is used to find 5' upstream sequence for all gene.
    :param up_length: the length of upstream sequence. Default=3000.
    :param rg_path: reference genome path. (dmel 6)
    :param gene_loci_file: gene location file
    :return: None.
    """

    # Find chromosome sequence
    reference_genome_files = os.listdir(rg_path)
    chr_dict = {}
    for file in reference_genome_files:
        chr_name = file[:-4]
        filename = rg_path + file
        with open(filename, 'r') as f:
            i = 0
            for line in f:
                if i == 0:
                    i += 1
                else:
                    chr_dict[chr_name] = line

    # Find gene upstream sequence
    sequence_dict = {}
    i = 0
    with open(gene_loci_file, 'r') as gene_file:
        gene_list = csv.reader(gene_file)
        for line in gene_list:
            if i == 0:
                i = 1
                continue
            gene_name = line[1]
            gene_loci = line[3]
            sequence = get_sequence(gene_loci, chr_dict, up_length)
            if sequence != "":
                sequence_dict[gene_name] = sequence

    # Delete sequence containing NA
    delete = []
    for key in sequence_dict:
        if 'N' in sequence_dict[key]:
            delete.append(key)
    for k in delete:
        sequence_dict.pop(k)

    # Convert to one hot
    one_hot = np.zeros([len(sequence_dict), 4, up_length, 1])
    j = 0
    for key in sequence_dict:
        sequence = sequence_dict[key]
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
        j += 1

    # Save data
    np.save(file="X.npy", arr=one_hot)
    with open("all_gene_sequence.json", "w") as json_file:
        json.dump(sequence_dict, json_file)


if __name__ == '__main__':
    get_upstream_sequence("./genome/", "./data/gene_loci.csv", )
