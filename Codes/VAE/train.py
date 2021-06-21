from VAE_dataset import DataSet
from torch.cuda.amp import GradScaler, autocast
from VAE import VAE
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import os
import pickle

if __name__ == '__main__':
    print(os.environ.get('CUDA_PATH'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Design model
    model = VAE()

    # Define hyper-parameters
    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-5

    # Check cuda
    print(torch.__version__)
    if torch.cuda.is_available():
        print("gpu ready")
        model.cuda()
    else:
        print("cuda not exist")

    # Prepare data
    train_dataset = DataSet('./data/')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # define loss_function
    reconstruction_function = nn.BCEWithLogitsLoss()


    def loss_function(recon_x, x, mu, logvar):
        """
        recon_x: generating DNA
        x: origin DNA
        mu: latent mean
        logvar: latent log variance
        """
        BCE = reconstruction_function(torch.flatten(recon_x, start_dim=1, end_dim=-1), torch.flatten(x, start_dim=1, end_dim=-1))  # bce loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.mean(KLD_element).mul_(-0.5)
        # KL divergence
        print(BCE)
        print(KLD)
        return BCE + KLD


    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Creates a GradScaler
    scaler = GradScaler()

    Loss = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_dataloader):
            print(batch_idx)
            data = data.type(torch.cuda.HalfTensor)
            data = data.to(device)
            optimizer.zero_grad()

            # Runs the forward pass with autocasting
            with autocast():
                recon_batch, mu, logvar= model(data)
                loss = loss_function(recon_batch, data, mu, logvar)

            scaler.scale(loss).backward()
            train_loss += loss.item()

            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader),
                    loss.item() / len(data)))

                # Save loss per 10 batch
                Loss.append(loss.item() / len(data))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset)))

        # Save model per epoch
        torch.save(model.state_dict(), './vae_epoch{}.pth'.format(
            epoch
        ))

        with open('loss.pkl', 'wb') as f:
            pickle.dump(Loss, f)

    torch.save(model.state_dict(), './vae.pth')
