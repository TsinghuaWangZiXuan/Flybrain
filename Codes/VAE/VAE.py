import torch
from torch import nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Flatten(nn.Module):
    """
        Flatten tensor to fit in fc_mu and fc_logvar layers
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """
        Unflatten tensor to fit in fc layer, default shape is (256, 1, 8, 4)
    """
    def forward(self, input):
        return input.view(input.size(0), 256, 1, 8, 4)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Inputs shape: (batch_size, 1, 32, 256, 128)

        self.encoder = nn.Sequential(

            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ELU(),

            nn.Conv3d(16, 16, 2, 2),
            nn.BatchNorm3d(16),
            nn.ELU(),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ELU(),

            nn.Conv3d(32, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.ELU(),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.Conv3d(64, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ELU(),

            nn.Conv3d(128, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.ELU(),

            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ELU(),

            nn.Conv3d(256, 256, 2, 2),
            nn.BatchNorm3d(256),
            nn.ELU(),

            Flatten()
        )

        self.fc_mu = nn.Linear(8192, 1024)
        self.fc_logvar = nn.Linear(8192, 1024)

        self.fc_decode = nn.Linear(1024, 8192)

        self.decoder = nn.Sequential(
            UnFlatten(),

            nn.ELU(),

            nn.ConvTranspose3d(256, 256, 2, 2),
            nn.BatchNorm3d(256),
            nn.ELU(),

            nn.ConvTranspose3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ELU(),

            nn.ConvTranspose3d(128, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.ELU(),

            nn.ConvTranspose3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.ConvTranspose3d(64, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.ConvTranspose3d(64, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ELU(),

            nn.ConvTranspose3d(32, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.ELU(),

            nn.ConvTranspose3d(32, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ELU(),

            nn.ConvTranspose3d(16, 16, 2, 2),
            nn.BatchNorm3d(16),
            nn.ELU(),

            nn.ConvTranspose3d(16, 1, 3, padding=1),

        )

    def encode(self, input):
        output = self.encoder(input)
        return self.fc_mu(output), self.fc_logvar(output)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        output = self.decoder(self.fc_decode(z))
        return output

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparametrize(mu, logvar)
        self.z = z
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def infer(self):
        """Fetches latent vector from model.

        :return: latent vector
        """
        return self.z
