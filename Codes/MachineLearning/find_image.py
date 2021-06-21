import os
import json
import csv
import pickle


def find_image(dna_list_file, images_path, map_file):
    """
    This function is used to find corresponding images.
    :param dna_list_file: whole dna file.
    :param images_path: file path of images.
    :param map_file: from DNA to images.
    :return: none.
    """
    with open(dna_list_file, 'r') as json_file:
        dna_list = json.load(json_file)

    file_names = os.listdir(images_path)
    num = 0
    dna_name = []
    nrrd_filename = []

    # VDRC data
    catalogue = open(map_file, 'r')
    reader = csv.reader(catalogue)
    i = 0
    for line in reader:
        if i == 0:
            i = 1
            continue
        file_name = line[0] + ".nrrd"
        if file_name in file_names and line[1][5:13] in dna_list and file_name not in nrrd_filename:
            dna_name.append(line[1][5:13])
            nrrd_filename.append(file_name)
            num = num + 1

    # Janelia data
    for file_name in file_names:
        if file_name[0] == 'G':
            name_list = file_name.split("_")
            name = name_list[1] + name_list[2]
            if name in dna_list:
                dna_name.append(name_list[1] + name_list[2])
                nrrd_filename.append(file_name)
                num = num + 1

    # Save data
    with open("./data/dna_name.pkl", 'wb') as df:
        pickle.dump(dna_name, df)

    with open("./data/nrrd_filename.pkl", 'wb') as nf:
        pickle.dump(nrrd_filename, nf)
    print(num)


if __name__ == '__main__':
    find_image("./data/dna_list.json",
               "F:/images",
               "./data/29417_List_all_images_included_in_Dickson_lab_VT_line_collection_-_VDRC_images.csv")
