from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from BERT import BERT
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PredictConcentrationModel(nn.Module):
    """
    predicting concentration
    """

    def __init__(self, hidden, max_len):
        """
        :param hidden: output size of BERT model
        :param max_len: max length of sentence
        """
        super().__init__()
        self.linear_1 = nn.Linear(hidden * max_len, 1024)
        self.linear_2 = nn.Linear(1024, 218)
        self.linear_3 = nn.Linear(218, 1)

    def forward(self, x):
        return self.linear_3(self.linear_2(self.linear_1(x)))


class BERTPCM(nn.Module):
    """
    BERT Predict Concentration Model
    """

    def __init__(self, bert):
        """
        :param bert: BERT model which should be trained
        """

        super().__init__()
        self.bert = bert
        self.pcm = PredictConcentrationModel(self.bert.hidden, self.bert.max_len)

    def forward(self, x):
        x = self.bert(x)
        x = x.view(-1, self.bert.max_len * self.bert.hidden)
        return self.pcm(x).view(-1)


class MyDataSet(Data.Dataset):
    def __init__(self, sequence, concentration):
        self.sequence = sequence
        self.concentration = concentration

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx], self.concentration[idx]


if __name__ == '__main__':
    # Check cuda
    print(torch.__version__)
    if torch.cuda.is_available():
        print("gpu ready")
    else:
        print("cuda not available")

    # Hyper parameters
    batch_size = 32
    epochs = 20
    n_layers = 12
    d_model = 768
    vocab_size = 5000
    max_len = 200
    n_heads = 12
    split_ratio = 0.9

    # Load data
    x = np.load("./data/X.npy")
    y = np.load("./data/Y.npy")

    # Transform Y
    scaler = StandardScaler()
    y = scaler.fit_transform(np.expand_dims(y, axis=-1))
    y = np.squeeze(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-split_ratio, shuffle=False)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=1 - split_ratio, random_state=42)

    loader = Data.DataLoader(MyDataSet(x_train, y_train), batch_size, True, num_workers=16)
    val_loader = Data.DataLoader(MyDataSet(x_validation, y_validation), batch_size, True, num_workers=16)

    # Load pre_trained model
    pre_model = torch.load('./model/bert.pth')
    bert_model = BERT(vocab_size, d_model, n_layers, n_heads, max_len)
    model_dict = bert_model.state_dict()
    state_dict = {k[5:]: v for k, v in pre_model.items() if k[5:] in model_dict.keys()}
    model_dict.update(state_dict)
    bert_model.load_state_dict(model_dict)

    # Freeze pre_trained model
    for param in bert_model.parameters():
        param.requires_grad = False

    model = BERTPCM(bert_model)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    model = model.to(device)

    losses = []
    validation_losses = []
    # Train model
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        batch_num = 0
        for batch_idx, [seq, conc] in enumerate(loader):
            seq = seq.to(device)
            conc = conc.float().to(device)

            optimizer.zero_grad()

            score = model(seq)
            print(score)
            print(conc)
            loss = criterion(score, conc)

            train_loss += loss.item()
            batch_num += 1

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(seq),
                    len(loader.dataset), 100. * batch_idx * len(seq) / len(loader.dataset),
                    loss.item()))

            loss.backward()
            optimizer.step()

        # Save model per epoch
        torch.save(model.state_dict(), './model/bert_pcm_epoch{}.pth'.format(
            epoch
        ))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / batch_num))
        losses.append(train_loss / batch_num)

        model.eval()
        val_loss = 0
        batch_num = 0
        with torch.no_grad():
            for batch_idx, [seq, conc] in enumerate(val_loader):
                seq = seq.to(device)
                conc = conc.to(device)
                score = model(seq)
                loss = criterion(score, conc)
                val_loss += loss.item()
                batch_num += 1
                if batch_idx % 10 == 0:
                    print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(seq),
                        len(val_loader.dataset), 100. * batch_idx * len(seq) / len(val_loader.dataset),
                        loss.item()))
        print('====> Epoch: {} Average val_loss: {:.4f}'.format(
            epoch, val_loss / batch_num))
        validation_losses.append(val_loss / batch_num)

    torch.save(model.state_dict(), './model/bert_pcm.pth')

    # Plot loss curve
    plt.plot(losses)
    plt.plot(validation_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
