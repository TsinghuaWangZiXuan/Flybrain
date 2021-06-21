import os
import numpy as np
from find_known_motif import find_known_motif


def make_pwm(motif_path, motif_max_length=21):
    """
    This function is used to make PWM files for training.
    :param motif_max_length: max length of tf motif, default is 21.
    :param motif_path: saved motif path.
    :return: None.
    """

    # Load tf motifs
    motif_files = os.listdir(motif_path)
    motif_num = len(motif_files)

    pwm = np.ones([4, motif_max_length, 1, 2 * motif_num])  # ACGT
    pwm /= 4  # padding is 0.25

    # Convert to PWM
    tf_list = []
    i = 0
    for file in motif_files:
        filename = motif_path + file
        symbol = file[:-4]
        tf_list.append(symbol)
        with open(filename, 'r') as f:
            data = f.readlines()
            j = 0
            reverse = []
            for line in data:
                line = line.split('\t')[:-1]
                line[0] = float(line[0])  # A
                line[1] = float(line[1])  # C
                line[2] = float(line[2])  # G
                line[3] = float(line[3])  # T
                reverse.append(line)

                pwm[0, j, 0, i] = line[0]  # A
                pwm[1, j, 0, i] = line[1]  # C
                pwm[2, j, 0, i] = line[2]  # G
                pwm[3, j, 0, i] = line[3]  # T
                j += 1

            # Reverse motif
            reverse = np.array(reverse)
            reverse = np.flip(reverse, axis=0)
            reverse = np.flip(reverse, axis=1)
            j = 0
            for line in reverse:
                # Same motif are at a distance of motif_num in vectors
                pwm[0, j, 0, i + motif_num] = line[0]  # A
                pwm[1, j, 0, i + motif_num] = line[1]  # C
                pwm[2, j, 0, i + motif_num] = line[2]  # G
                pwm[3, j, 0, i + motif_num] = line[3]  # T
                j += 1
        i += 1

    np.save("PWM.npy", pwm)


if __name__ == '__main__':
    make_pwm("./motif/",
             find_known_motif("DC3_TF.txt",
                              "./data/Fly TFs in brain and VNC.csv",
                              "./data/fly_factor_survey.meme",
                              "./motif/"))
