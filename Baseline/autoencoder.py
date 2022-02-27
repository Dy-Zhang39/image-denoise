import numpy as np
import time
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Training import utility
from Training.dataloader import get_dataloaders


class Autoencoder_cnn(nn.Module):
    def __init__(self):
        super(Autoencoder_cnn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embed(self, x):
        return self.encoder(x)

    def decode(self, e):
        return self.decoder(e)


def train(model, batch_size=256, num_epochs=1, learning_rate = 0.01):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iters, losses = [], []
    n = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        mini_b = 0
        for dirty, clean in train_loader:
            dirty, clean = utility.normalization(dirty, clean)
            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                dirty = dirty.cuda()
                clean = clean.cuda()
            #############################################

            out = model(dirty)  # forward pass

            loss = criterion(out, clean)  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch

            iters.append(n)
            losses.append(float(loss) / batch_size)  # compute *average* loss

            '''
            image = out[0].cpu()
            image = image.detach().numpy()
            img = np.transpose(image, [1, 2, 0])
            plt.axis('off')
            plt.title('denoised')
            plt.imshow(img)
            plt.show()
            '''

            n += 1
            mini_b += 1
            print("Iteration: ", n,
                  'Progress: % 6.2f ' % ((epoch * len(train_loader) + mini_b) / (num_epochs * len(train_loader)) * 100),
                  '%', "Time Elapsed: % 6.2f s " % (time.time() - start_time))

    end_time = time.time()

    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    print("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % (
        (end_time - start_time), ((end_time - start_time) / num_epochs)))

if __name__ == '__main__':
    use_cuda = True
    num_workers = 0
    batch_size = 256
    train_loader, val_loader, test_loader = get_dataloaders(train_path="../Dataset/Merged_Dataset/train",
                                                            val_path="../Dataset/Merged_Dataset/val",
                                                            test_path="../Dataset/Merged_Dataset/test",

                                                            batch_size=batch_size)
    #baseline model
    batch_size = 256
    model = Autoencoder_cnn()

    if use_cuda and torch.cuda.is_available():
        model.cuda()
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')



    train(model, batch_size=batch_size, num_epochs=10)

    k = 0
    model.cpu()
    use_cuda = False
    for dirty, clean in test_loader:
        dirty, clean = utility.normalization(dirty, clean)

        out = model(dirty)
        image = dirty[0]
        # place the colour channel at the end, instead of at the beginning
        img = np.transpose(image, [1, 2, 0])
        plt.subplot(6, 3, k + 1)
        plt.axis('off')
        plt.title('dirty')
        plt.imshow(img)

        image = out[0]
        image = image.detach().numpy()
        img = np.transpose(image, [1, 2, 0])
        plt.subplot(6, 3, k + 2)
        plt.axis('off')
        plt.title('denoised')
        plt.imshow(img)

        image = clean[0]
        # place the colour channel at the end, instead of at the beginning
        img = np.transpose(image, [1, 2, 0])
        plt.subplot(6, 3, k + 3)
        plt.axis('off')
        plt.title('clean')
        plt.imshow(img)

        print(dirty[0])
        print(out[0])
        print(clean[0])
        k += 3
        if k > 10:
            break
    plt.show()


