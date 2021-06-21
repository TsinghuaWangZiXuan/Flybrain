from VAE_dataset import DataSet
from VAE import VAE
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

if __name__ == '__main__':
    # Design model
    model = VAE()
    model.load_state_dict(torch.load("vae.pth"))

    if torch.cuda.is_available():
        print("gpu ready")
        model.cuda()
        print(torch.__version__)

    # Prepare data
    path = './data/'
    test_dataset = DataSet(path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)

    # Test
    latent_vector = []
    model.eval()
    with torch.no_grad():
        for i, test_sample in enumerate(test_loader):
            print("{} step".format(i))
            test_sample = test_sample.type(torch.cuda.FloatTensor)
            test_sample = test_sample.to('cuda')
            recon_batch, m, l = model(test_sample)
            recon_batch = torch.nn.Sigmoid()(recon_batch)
            recon_batch = recon_batch.cuda().data.cpu().numpy()

            recon_batch = np.squeeze(recon_batch)
            recon_batch = recon_batch*255
            print(np.max(recon_batch))

            # Convert to images
            for j in range(recon_batch.shape[0]):
                image = np.transpose(recon_batch[j, :, :])
                Image.fromarray(image).save("./image/{}_{}.tif".format(
                    i,
                    j
                ))

            # Infer latent vector
            z = model.infer()

            z = np.squeeze(z)

            z = z.tolist()
            latent_vector.append(z)
