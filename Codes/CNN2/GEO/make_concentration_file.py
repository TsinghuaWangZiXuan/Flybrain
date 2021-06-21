import os
import numpy as np


def make_concentration_file(tf_concentration_file, motif_path):
    """
    This function is used to make tf concentration file for training.
    :param tf_concentration_file: DC3 adPN tf concentration file.
    :param motif_path: saved motif path.
    :return: None
    """

    # Load tf symbol
    tf_file = open(tf_concentration_file, 'r')
    motif_file_name = os.listdir(motif_path)
    tf_symbol = []
    for file in motif_file_name:
        tf_symbol.append(file[:-4])

    # Load concentration data
    tf_concentration = {}
    for line in tf_file:
        line = line.split('\t')[:-1]
        tf_concentration[line[0]] = float(line[1]) / 10  # Concentration data is scaled to 1

    # Find concentration of candidate tf
    concentration = []
    for s in tf_symbol:
        concentration.append(tf_concentration[s])
    concentration = np.array(concentration)
    np.save(arr=concentration, file="conc.npy")


if __name__ == '__main__':
    make_concentration_file("DC3_TF.txt", "./motif/")
