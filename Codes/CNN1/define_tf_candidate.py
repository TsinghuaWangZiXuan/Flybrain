import csv


def define_tf_candidate(brain_tf_file, fly_factor_survey, out_path):
    """
    This function is used to define tf candidate for training. Here we choose hox tf.
    :param brain_tf_file: important tf in fly brain.
    :param fly_factor_survey: fly factor survey dataset.
    :param out_path: path for save tf motif files.
    :return: max length of motifs.
    """

    # Load tf file
    f = open(brain_tf_file, 'r')
    file = csv.reader(f)

    # Find hox tf
    i = 0
    hox_id2symbol = {}
    for Line in file:
        if i == 0:
            i = i + 1
            continue
        symbol = Line[7]
        if symbol == '1' or symbol == '0':
            hox_id2symbol[Line[2]] = Line[10]

    # Find motif of candidate tf
    length = []
    with open(fly_factor_survey, 'r') as meme:
        all_lines = meme.readlines()
        i = 0
        for line in all_lines:
            ind = line.split(' ')[0]
            if ind == 'MOTIF':
                motif_id = line.split(' ')[1]
                if motif_id in hox_id2symbol:
                    symbol = hox_id2symbol[motif_id]
                    pwm = []
                    hox_info = all_lines[i + 2].split(' ')
                    motif_length = int(hox_info[5])
                    length.append(motif_length)
                    for j in range(motif_length):
                        pwm.append(all_lines[i + 3 + j])
                    with open(out_path + symbol + ".txt", 'w') as outfile:
                        outfile.writelines(pwm)
            i = i + 1

    print("Max length of motifs is {}".format(max(length)))
    return max(length)


if __name__ == '__main__':
    define_tf_candidate('./data/Fly TFs in brain and VNC.csv', './data/fly_factor_survey.meme', "./motif/")
