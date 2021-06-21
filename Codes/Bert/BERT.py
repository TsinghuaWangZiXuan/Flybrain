import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tokenization import tokenization
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_data(data, vocab_size, max_len):
    """
    This function makes data for training from DNA file directly.
    """
    batch = []
    # Crop sequence
    for dna in data:
        input_ids = dna
        if len(dna) > max_len:
            n = len(dna) - max_len
            input_ids = dna[n:]

        # MASK LM
        n_pred = int(len(input_ids) * 0.15)  # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)]  # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens = list(input_ids)
        for pos in cand_maked_pos[:n_pred]:
            if random() < 0.8:  # 80%
                input_ids[pos] = 1  # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                while index < 3:
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # replace

        # Zero Paddings
        if len(input_ids) < max_len:
            n_pad = max_len - len(input_ids)
            input_ids.extend([0] * n_pad)
            masked_tokens.extend([0] * n_pad)
        batch.append([input_ids, masked_tokens])

    return batch


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx]


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden, n_layers, attn_heads, max_len, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.max_len = max_len
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, max_len=max_len)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        return self.mask_lm(x)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


if __name__ == '__main__':
    # Check cuda
    print(torch.__version__)
    if torch.cuda.is_available():
        print("gpu ready")
    else:
        print("cuda not available")

    # Hyper parameters
    batch_size = 16
    epochs = 20
    n_layers = 12
    d_model = 768
    vocab_size = 5000
    max_len = 200
    n_heads = 12
    split_ratio = 0.9

    # Load tokenized data
    dna_file = "./data/all_sequence.txt"
    data = tokenization('train', dna_file)
    shuffle(data)

    # Define model
    model = BERTLM(BERT(vocab_size, d_model, n_layers, n_heads, max_len), 5000)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Learning rate warm up
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10000, max_iters=2000000)

    model = model.to(device)

    losses = []
    validation_losses = []
    # Train model
    for epoch in range(epochs):
        # Dynamic masking
        # Make data for training
        batch = make_data(data, vocab_size, max_len)
        input_ids, masked_tokens = zip(*(batch[:int(len(batch) * 0.9)]))
        input_ids, masked_tokens = torch.LongTensor(input_ids), torch.LongTensor(masked_tokens)

        # Make data for validation
        val_input_ids, val_masked_tokens = zip(*(batch[int(len(batch) * 0.9):]))
        val_input_ids, val_masked_tokens = torch.LongTensor(val_input_ids), torch.LongTensor(val_masked_tokens)

        # Define data loader
        loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens), batch_size, True, num_workers=16)
        val_loader = Data.DataLoader(MyDataSet(val_input_ids, val_masked_tokens), batch_size, True, num_workers=16)

        model.train()
        train_loss = 0

        batch_num = 0
        for batch_idx, [input_ids, masked_tokens] in enumerate(loader):
            # input_ids = input_ids.type(torch.cuda.HalfTensor)
            input_ids = input_ids.to(device)
            # masked_tokens = masked_tokens.type(torch.cuda.HalfTensor)
            masked_tokens = masked_tokens.to(device)

            optimizer.zero_grad()

            logits_lm = model(input_ids)
            loss = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM

            train_loss += loss.item()
            batch_num += 1

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(input_ids),
                    len(loader.dataset), 100. * batch_idx * len(input_ids) / len(loader.dataset),
                    loss.item()))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # Save model per epoch
        torch.save(model.state_dict(), './model/bert_epoch{}.pth'.format(
            epoch
        ))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / batch_num))
        losses.append(train_loss / batch_num)

        model.eval()
        val_loss = 0
        batch_num = 0
        with torch.no_grad():
            for batch_idx, [input_ids, masked_tokens] in enumerate(val_loader):
                input_ids = input_ids.to(device)
                masked_tokens = masked_tokens.to(device)
                logits_lm = model(input_ids)
                loss = criterion(logits_lm.transpose(1, 2), masked_tokens)
                val_loss += loss.item()
                batch_num += 1
                if batch_idx % 10 == 0:
                    print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(input_ids),
                        len(val_loader.dataset), 100. * batch_idx * len(input_ids) / len(val_loader.dataset),
                        loss.item()))
        print('====> Epoch: {} Average val_loss: {:.4f}'.format(
            epoch, val_loss / batch_num))
        validation_losses.append(val_loss / batch_num)

    torch.save(model.state_dict(), './model/bert.pth')

    # Plot loss curve
    plt.plot(losses)
    plt.plot(validation_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
