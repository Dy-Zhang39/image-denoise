import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim #for gradient descent
import torchvision

from Training.dataloader import get_dataloaders
import utility


class CBDnet(nn.Module):
    # please implement CBDnet here, this is just return the same size image
    def __init__(self):
        super(CBDnet, self).__init__()
        #input image is 3 * 256 * 256
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (256+2*1-3)/1+1=256
        return x

''' redundent
def get_accuracy(model, train=0):
    if train == 0:
        data_loader = train_loader
    elif train == 1:
        data_loader = val_loader
    else:
        data_loader = test_loader

    correct = 0
    total = 0
    for imgs, labels in data_loader:

        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################

        output = model(imgs)

        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total
'''

def train(model, batch_size=20, num_epochs=1, learning_rate=0.01, train_type=0):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0  # the number of iterations
    start_time = time.time()
    for epoch in range(num_epochs):
        mini_b = 0
        mini_batch_correct = 0
        Mini_batch_total = 0
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

            ##### Mini_batch Accuracy ##### We don't compute accuracy on the whole trainig set in every iteration!
            #pred = out.max(1, keepdim=True)[1]
            #mini_batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            #Mini_batch_total = imgs.shape[0]
            #train_acc.append((mini_batch_correct / Mini_batch_total))
            ###########################

            # save the current training information
            iters.append(n)
            losses.append(float(loss) / batch_size)  # compute *average* loss
            #val_acc.append(get_accuracy(model, train=train_type + 1))  # compute validation accuracy
            n += 1
            mini_b += 1
            print("Iteration: ", n,
                  'Progress: % 6.2f ' % ((epoch * len(train_loader) + mini_b) / (num_epochs * len(train_loader)) * 100),
                  '%', "Time Elapsed: % 6.2f s " % (time.time() - start_time))

        print("Epoch %d Finished. " % epoch, "Time per Epoch: % 6.2f s " % ((time.time() - start_time) / (epoch + 1)))

    end_time = time.time()

    utility.data_plotting(iters, losses)

    print("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % (
    (end_time - start_time), ((end_time - start_time) / num_epochs)))


if __name__ == '__main__':
    use_cuda = True
    num_workers=0

    train_loader, val_loader, test_loader = get_dataloaders(train_path="../Dataset/Merged_Dataset/train",
                                                            val_path="../Dataset/Merged_Dataset/val",
                                                            test_path="../Dataset/Merged_Dataset/test",
                                                            batch_size=30)

    # proper model
    batch_size = 256
    model = CBDnet()

    if use_cuda and torch.cuda.is_available():
        model.cuda()
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')

    train(model, batch_size=batch_size, num_epochs=10)