import numpy as np
import time
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# from GPUtil import showUtilization as gpu_usage  # conda install -c conda-forge gputil
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
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3),
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


def train(model, batch_size=256, num_epochs=1, learning_rate = 1e-4):
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


def get_example(model, data_loader, batch_size):
    k = 0
    for dirty, clean in data_loader:
        dirty, clean = utility.normalization(dirty, clean)
        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            dirty = dirty.cuda()
            clean = clean.cuda()
        #############################################

        for i in range(batch_size):
            out = model(dirty)
            image = dirty.cpu()[i]
            # place the colour channel at the end, instead of at the beginning
            img = np.transpose(image, [1, 2, 0])
            plt.subplot(6, 3, k + 1)
            plt.axis('off')
            plt.title('dirty')
            plt.imshow(img)

            image = out.cpu()[i]
            image = image.detach().numpy()
            img = np.transpose(image, [1, 2, 0])
            plt.subplot(6, 3, k + 2)
            plt.axis('off')
            plt.title('denoised')
            plt.imshow(img)

            image = clean.cpu()[i]
            # place the colour channel at the end, instead of at the beginning
            img = np.transpose(image, [1, 2, 0])
            plt.subplot(6, 3, k + 3)
            plt.axis('off')
            plt.title('clean')
            plt.imshow(img)

            k += 3
            if k > 10:
                break

        if k > 10:
            break
    plt.show()




if __name__ == '__main__':
    use_cuda = True
    num_workers = 0
    batch_size = 64
    train_loader, val_loader, test_loader = get_dataloaders(train_path="../Dataset/Merged_Dataset/train",
                                                            val_path="../Dataset/Merged_Dataset/val",
                                                            test_path="../Dataset/Merged_Dataset/test",

                                                            batch_size=batch_size)
    #baseline model
    model = Autoencoder_cnn()

    if use_cuda and torch.cuda.is_available():
        model.cuda()
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')

    train(model, batch_size=batch_size, num_epochs=30)

    #get_example(model, data_loader=val_loader, batch_size=batch_size)
    #get_example(model, data_loader=train_loader, batch_size=batch_size)
    get_example(model, data_loader=test_loader, batch_size=batch_size)
    count = utility.save_model_output(model, use_cuda)

    path = "../Baseline/new model parameter"
    torch.save(model.state_dict(), path)
    print("Reality PSNR is {}".format(utility.PSNR(model, count, True)))
    print("Reality SSIM is {}".format(utility.SSIM(model, count, True)))

    print("Baseline Model PSNR is {}".format(utility.PSNR(model, count, False)))
    print("Baseline Model SSIM is {}".format(utility.SSIM(model, count, False)))

