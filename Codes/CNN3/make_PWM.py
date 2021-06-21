import numpy as np


def make_pwm(fly_factor_survey, tf_id_list, max_length=21):
    """
    This function is used to get PWM from fly factor survey data.
    :param fly_factor_survey: fly_factor_survey dataset.
    :param tf_id_list: Target TF ID list
    :param max_length: max length of motif
    :return: pwm of target tf
    """
    pwm = np.ones([4, max_length, 1, 2 * len(tf_id_list)], dtype=float)  # ACGT
    pwm /= 4

    with open(fly_factor_survey, 'r') as meme:
        all_lines = meme.readlines()
        i = 0
        known_tf_num = 0
        for line in all_lines:
            ind = line.split(' ')[0]
            if ind == 'MOTIF':
                motif_name = line.split(' ')[1]
                if motif_name in tf_id_list:
                    hox_info = all_lines[i + 2].split(' ')
                    tf_length = int(hox_info[5])
                    reverse = []
                    for j in range(tf_length):
                        site = all_lines[i + 3 + j]
                        site = site.split('\t')[:-1]
                        site[0] = float(site[0])  # A
                        site[1] = float(site[1])  # C
                        site[2] = float(site[2])  # G
                        site[3] = float(site[3])  # T
                        reverse.append(site)

                        pwm[0, j, 0, known_tf_num] = site[0]  # A
                        pwm[1, j, 0, known_tf_num] = site[1]  # C
                        pwm[2, j, 0, known_tf_num] = site[2]  # G
                        pwm[3, j, 0, known_tf_num] = site[3]  # T

                    reverse = np.array(reverse)
                    reverse = np.flip(reverse, axis=0)
                    reverse = np.flip(reverse, axis=1)

                    j = 0
                    for site in reverse:
                        pwm[0, j, 0, known_tf_num + len(tf_id_list)] = site[0]  # A
                        pwm[1, j, 0, known_tf_num + len(tf_id_list)] = site[1]  # C
                        pwm[2, j, 0, known_tf_num + len(tf_id_list)] = site[2]  # G
                        pwm[3, j, 0, known_tf_num + len(tf_id_list)] = site[3]  # T
                        j += 1
                    known_tf_num += 1
            i = i + 1

    return pwm
