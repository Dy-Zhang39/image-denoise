import time
import cv2 #pip3 install opencv-contrib-python
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim #for gradient descent
import torchvision

from Training.dataloader import get_dataloaders
from Training import utility

# should add dropout in the future
class CBDnet(nn.Module):
    # please implement CBDnet here, this is just return the same size image
    def __init__(self):
        super(CBDnet, self).__init__()
        #input image is 3 * 256 * 256
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (256+2*1-3)/1+1=256
        return x


# the training code is adapted from tut 3a, adding data normalization and weight decay to prevent overfitting
def train(model, batch_size=20, num_epochs=1, learning_rate=0.01, train_type=0, weight_decay = 0.001):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0  # the number of iterations
    start_time = time.time()
    for epoch in range(num_epochs):
        mini_b = 0
        for imgs, labels in iter(train_loader):
            imgs,labels = utility.normalization(imgs, labels)

            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################

            #update
            out = model(imgs)  # forward pass

            loss = criterion(out, labels)  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch


            # save the current training information
            iters.append(n)
            losses.append(float(loss) / batch_size)  # compute *average* loss

            n += 1
            mini_b += 1
            print("Iteration: ", n,
                  'Progress: % 6.2f ' % ((epoch * len(train_loader) + mini_b) / (num_epochs * len(train_loader)) * 100),
                  '%', "Time Elapsed: % 6.2f s " % (time.time() - start_time))

        print("Epoch %d Finished. " % epoch, "Time per Epoch: % 6.2f s " % ((time.time() - start_time) / (epoch + 1)))

    path = "model parameter"
    torch.save(model.state_dict(), path)

    end_time = time.time()

    utility.loss_plotting(iters, losses)

    print("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % (
    (end_time - start_time), ((end_time - start_time) / num_epochs)))


if __name__ == '__main__':
    use_cuda = True
    num_workers=0
    batch_size = 256


    train_loader, val_loader, test_loader = get_dataloaders(train_path="../Dataset/Merged_Dataset/train",
                                                            val_path="../Dataset/Merged_Dataset/val",
                                                            test_path="../Dataset/Merged_Dataset/test",
                                                            batch_size=30)

    # proper model
    train_model = False # if we need to train the model

    model = utility.load_model(train_model)

    if use_cuda and torch.cuda.is_available():
        model.cuda()
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')


    if (train_model):
        train(model, batch_size=batch_size, num_epochs=10)

    count = utility.save_model_output(model, use_cuda)

    print("The Average PSNR between model prediction and clean is {}".format(utility.PSNR(model,count,psnr_predict_clean= True)))
    print("The Average SSIM between model prediction and clean is {}".format(utility.SSIM(model,count,psnr_predict_clean= True)))

    print("The Average PSNR has improved by {}".format(
        utility.PSNR(model, count, psnr_predict_clean=True) - utility.PSNR(model, count, psnr_predict_clean=False)))
    print("The Average SSIM has improved by {}".format(
        utility.SSIM(model, count, psnr_predict_clean=True) - utility.SSIM(model, count, psnr_predict_clean=False)))

