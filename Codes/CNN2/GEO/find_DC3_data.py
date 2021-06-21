import csv
import pickle


def find_dc3_data(scRNA_file, cells_file, brain_tf_file):
    """
    This function is used to find tf expression in DC3 adPN(72h APF).
    :param scRNA_file: single cell sequencing file, format is tab.
    :param cells_file: DC3 adPN cells file.
    :param brain_tf_file: important tf in fly brain.
    :return: None.
    """

    # Get gene list from first column of scRNA file
    genes = []
    with open(scRNA_file, 'r') as f:
        f = f.readlines()
        for line in f:
            line = line[:-1]
            line = line.split('\t')
            genes.append(line)

    # Get cell name of dc3
    dc3 = []
    dc3_name = []
    index = [0]
    with open(cells_file, 'r') as f:
        f = f.readlines()
        for line in f:
            line = line[:-1]
            dc3_name.append(line)
        for i in range(len(genes[0])):
            if genes[0][i] in dc3_name:
                index.append(i)
        for Line in genes:
            l = []
            for item in index:
                l.append(Line[item])
            dc3.append(l)

    # Get brain tf concentration from scRNA data
    with open(brain_tf_file, "r") as f:
        f = csv.reader(f)
        tf = []
        for line in f:
            tf.append(line[10])
    dc3_tf = [dc3[0]]
    mean = []
    for gene in dc3:
        if gene[0] in tf:
            n = 0
            sum = 0
            for i in gene:
                if n == 0:
                    n = 1
                    continue
                sum += float(i)
            if sum > 0:
                dc3_tf.append(gene)
                mean.append(sum / (len(gene) - 1))
    dc3 = []
    i = 0
    for line in dc3_tf:
        if line[0] == "symbol":
            continue
        else:
            dc3.append([line[0], mean[i]])
            i += 1

    # Sort data by descending order
    def sec(a):
        return a[1]
    dc3.sort(key=sec, reverse=True)
    print(dc3)

    # Output
    with open("dc3_tf.pkl", 'wb') as f:
        pickle.dump(dc3_tf, f)

    with open("dc3_tf.txt", 'w') as f:
        for line in dc3:
            for i in line:
                f.write(str(i))
                f.write("\t")
            f.write('\n')


if __name__ == '__main__':
    find_dc3_data("./data/htseq_logCPM_neurons.tab", "./data/names_DC3_72hAPF.txt", "./data/Fly TFs in brain and VNC.csv")
