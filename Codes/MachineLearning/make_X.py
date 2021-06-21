import json
import numpy as np
import pickle


def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


if __name__ == '__main__':
    with open("./data/Janelia_DNA.json", "r") as jf:
        j_dna = json.load(jf)
    with open("./data/VDRC_DNA.json", "r") as vf:
        v_dna = json.load(vf)
    with open("./data/dna_name.pkl", "rb") as hf:
        dna_name = pickle.load(hf)

    # Merge two data sets
    all_dna = merge(j_dna, v_dna)

    x = np.zeros((len(dna_name), 4, 3000), dtype=None)  # default length is 3000

    i = 0

    for key in dna_name:
        x[i] = all_dna[key]
        i = i + 1
        print(i)

    np.save(file="./data/X.npy", arr=x)
