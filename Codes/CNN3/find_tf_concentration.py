import numpy as np
import pickle
import csv
from make_PWM import make_pwm


def find_tf_concentration(concentration_file, selected_gene_file, known_tf_file, fly_factor_survey, brain_tf_file):
    """
    This function is used to find hox tf concentration for training.
    :param concentration_file: all gene concentration.
    :param selected_gene_file: genes that has concentration information.
    :param known_tf_file: hox tf id whose motifs are known.
    :param fly_factor_survey: fly_factor_survey dataset.
    :param brain_tf_file: important tf in brain.
    :return: None
    """

    # Load concentration
    concentration = np.load(concentration_file)
    file = open(selected_gene_file, 'rb')
    gene_list = pickle.load(file)

    # Load tf whose motifs are known
    file = open(known_tf_file, 'rb')
    known_tf = pickle.load(file)

    # Load all tf
    hox = []
    sym2id = {}
    with open(brain_tf_file, 'r') as f:
        tf = csv.reader(f)
        for line in tf:
            sym2id[line[10]] = line[2]
            if line[7] == '1' and line[2] in known_tf:
                hox.append(line[10])

    # Find tf concentration
    tf_concentration = {}
    for gene in gene_list:
        if gene in hox:
            if concentration[gene_list.index(gene)] > 0:
                tf_concentration[gene] = concentration[gene_list.index(gene)]

    tf_concentration = sorted(tf_concentration.items(), key=lambda d: d[1], reverse=True)

    tf_list = []
    concentration = []
    for hox in tf_concentration:
        id = sym2id[hox[0]]
        tf_list.append(id)
        concentration.append(hox[1])

    # Make pwm
    pwm = make_pwm(fly_factor_survey, tf_list)
    concentration = np.asarray(concentration, dtype=float)
    np.save(arr=pwm, file='./data/pwm.npy')
    np.save(arr=concentration, file='./data/conc.npy')


if __name__ == '__main__':
    find_tf_concentration("./data/Y.npy", "./data/selected_gene.pkl", "./data/known_tf.pkl",
                          "./data/fly_factor_survey.meme", "./data/Fly TFs in brain and VNC.csv")
