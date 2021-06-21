import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.fc1 = nn.Linear(2000, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, 1024)
        self.fc8 = nn.Linear(1024, 2000)

        self.softmax = nn.Softmax(dim=1)

    def encode(self, x):
        h1 = torch.flatten(x, start_dim=1, end_dim=-1)
        return self.fc3(self.fc2(self.fc1(h1)))

    def decode(self, z):
        h3 = self.fc8(self.fc7(self.fc6(z)))
        h3 = torch.reshape(h3, [-1, 4, 500])
        return self.softmax(h3)

    def forward(self, x):
        z = self.encode(x)
        self.z = z
        return self.decode(z)

    @torch.no_grad()
    def infer(self):
        """Fetches latent vector from model.

        :return: latent vector
        """
        return self.z
