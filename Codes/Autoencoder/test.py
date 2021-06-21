from DataSet import MyDataset
from Autoencoder import Autoencoder
import torch
from torch.utils.data import DataLoader
import numpy as np


def accuracy(y_pre, y_true):
    y_true = np.reshape(y_true, (y_true.shape[0] * y_true.shape[1]))

    for k in range(y_pre.shape[1]):
        m = np.max(y_pre[:, k])
        state = 0
        for l in range(4):
            if y_pre[l, k] == m and state != 1:
                y_pre[l, k] = 1
                state = 1
            else:
                y_pre[l, k] = 0
    y_pre = np.reshape(y_pre, (y_pre.shape[0] * y_pre.shape[1]))

    num = np.dot(y_pre, y_true)
    return num/500


if __name__ == '__main__':


    # Design model
    model = Autoencoder()

    # Load parameters
    model.load_state_dict(torch.load("autoencoder.pth"))

    # Prepare data
    file = open('AllDNA.txt', 'r')
    dna = file.readlines()
    test_dataset = MyDataset(dna, 500)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    latent_vector = []

    model.eval()
    torch.no_grad()
    for i, test_data in enumerate(test_loader):
        print("{} step".format(i))
        test_data = test_data.to(torch.float32)

        recon_batch = model(test_data)

        recon_batch = recon_batch.detach().numpy()
        test_data = test_data.detach().numpy()
        recon_batch = np.squeeze(recon_batch)
        test_data = np.squeeze(test_data)
        acc = accuracy(recon_batch, test_data)

        print(acc)
        # z = model.infer()
        # z = z.cuda().data.cpu().numpy()

        # latent_vector.append(z)

    # latent_vector = np.concatenate(latent_vector, axis=0)
    # np.save(arr=latent_vector, file="latent_vector.npy")
