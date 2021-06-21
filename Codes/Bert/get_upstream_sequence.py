import csv
import os
import json


def flip_sequence(dna_seq):
    reverse_seq = list(dna_seq[::-1])
    for i, base in enumerate(reverse_seq):
        if base == 'A':
            reverse_seq[i] = 'T'
        if base == 'C':
            reverse_seq[i] = 'G'
        if base == 'G':
            reverse_seq[i] = 'C'
        if base == 'T':
            reverse_seq[i] = 'A'
    reverse_seq = "".join(reverse_seq)
    return reverse_seq


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


def get_upstream_sequence(rg_path, gene_loci_file, up_length=1000):
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
            rev_sequence = flip_sequence(sequence)
            if sequence != "":
                sequence_dict[gene_name] = sequence
                sequence_dict[gene_name + '_reverse'] = rev_sequence

    # Save data
    with open("./data/all_gene_sequence.json", "w") as json_file:
        json.dump(sequence_dict, json_file)


if __name__ == '__main__':
    get_upstream_sequence("./genome/", "./data/gene_loci.csv", )
