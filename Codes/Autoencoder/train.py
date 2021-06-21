import pickle
from torch.cuda.amp import GradScaler, autocast
from Autoencoder import Autoencoder
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from DataSet import MyDataset


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Design model
    model = Autoencoder()
    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-3

    # Check cuda
    if torch.cuda.is_available():
        print("gpu ready")
        model.cuda()
        print(torch.__version__)
    else:
        print("need cuda")

    # Prepare data
    file = open('./data/AllDNA.txt', 'r')
    dna = file.readlines()
    # PWM=np.load(file='PWM.npy')
    train_dataset = MyDataset(dna, 500)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # Define loss function
    reconstruction_function = nn.MSELoss()


    def loss_function(recon_x, x):
        BCE = reconstruction_function(torch.flatten(recon_x, start_dim=1, end_dim=-1),
                                      torch.flatten(x, start_dim=1, end_dim=-1))  # mse loss
        return BCE


    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Creates a GradScaler
    scaler = GradScaler()

    Loss = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # Runs the forward pass with autocasting
            with autocast():
                recon_batch = model(data)
                loss = loss_function(recon_batch, data)

            scaler.scale(loss).backward()
            train_loss += loss.item()

            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

                # Save loss per 10 batch
                Loss.append(loss.item())

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

        # Save model per epoch
        torch.save(model.state_dict(), './autoencoder_epoch{}.pth'.format(
            epoch
        ))

        with open('loss.pkl', 'wb') as f:
            pickle.dump(Loss, f)

    torch.save(model.state_dict(), './autoencoder.pth')
