from Bio import SeqIO
from collections import defaultdict


def flank_sequence(bed_file, fasta_file, out_file):
    """
    This function is used to get flank_sequence of genes from bed file based on fasta file.
    """
    # read names and postions from bed file
    positions = defaultdict(list)

    bf = open(bed_file, 'r')
    for line in bf:
        name, start, stop, gene_name = line.strip().split()
        positions[name].append((int(start), int(stop)))

    # parse fasta file and turn into dictionary
    file = open(fasta_file, 'r')
    records = SeqIO.parse(file, 'fasta')
    records = SeqIO.to_dict(records)

    sum = 0
    # search for short sequences
    short_seq_records = []
    for name in positions:
        long_seq_record = records[name]
        long_seq = long_seq_record.seq
        for (start, stop) in positions[name]:
            short_seq = str(long_seq)[start:stop]
            short_seq = short_seq.upper()
            if len(short_seq) == 1000:
                sum += 1
                print(sum)
                short_seq_records.append(short_seq)

    # write to file
    with open(out_file, 'w') as f:
        f.write('\n'.join(short_seq_records))

    return short_seq_records


if __name__ == '__main__':
    species = "rat"
    flank_sequence("./data/bed/{}_regulatory_sequence.bed".format(species),
                   "./data/fasta/{}_genome.fasta".format(species),
                   "./data/seq/{}_sequence.txt".format(species))
