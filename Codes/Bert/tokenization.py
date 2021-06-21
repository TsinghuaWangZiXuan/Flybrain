import sentencepiece as spm


def tokenization(mode, dna_file=None):
    if mode == 'build':
        sp = spm.SentencePieceTrainer
        sp.Train(input='./data/all_gene_sequence.txt',
                 vocab_size=5000,
                 model_prefix='./model/mypiece',
                 model_type='bpe')
    elif mode == 'train':
        sp = spm.SentencePieceProcessor(model_file='./model/mypiece.model')
        seq = open(dna_file, 'r')
        tokens_list = []
        for line in seq:
            tokens = sp.encode(line[:-1])
            tokens_list.append(tokens)
        return tokens_list
    elif mode == 'test':
        sp = spm.SentencePieceProcessor(model_file='./model/mypiece.model')
        print(sp.decode([4999]))


if __name__ == '__main__':
    tokenization('test')
