import os
import time
import cv2  # pip3 install opencv-contrib-python
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim  # for gradient descent
import torchvision

from dataloader import get_dataloaders
import utility


##########################################################################################
# should add dropout in the future
# The following code is adapted from the following website: https://github.dev/IDKiro/CBDNet-pytorch

##########################################################################################
# model 0 low complexity
#
class CBDnet_0(nn.Module):
    # please implement CBDnet here, this is just return the same size image
    def __init__(self):
        super(CBDnet_0, self).__init__()
        #input image is 3 * 256 * 256
        self.fcn = FCN_0()            #CNN_E: takes an noisy observtion y and output esitmate the noise level map
        self.unet = UNet_0()          #UNet: performs image denoise

        #daniel#self.conv1 = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        noise_level = self.fcn(x)           #get the noise level map of input x
        concat_img = torch.cat([x, noise_level], dim=1)         #combine the two tensor together as an inout to unet
        out = self.unet(concat_img) + x     #taking both noisy image and noise level map as input is helpful in generalizing the learned model to images beyond the noise model
        return out
        #daniel#x = F.relu(self.conv1(x)) # (256+2*1-3)/1+1=256
        #return x

#FCN class estimate noise level of the input signal
class FCN_0(nn.Module):  # CNN_E
    def __init__(self):
        super(FCN_0, self).__init__()
        self.fcn = nn.Sequential(           #5 CNN with channel size all set to 32
            nn.Conv2d(3, 32, 3, padding=1),     #conv2d ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fcn(x)

#UNet perform denoise through three levels
class UNet_0(nn.Module):
    def __init__(self):
        super(UNet_0, self).__init__()

        self.inc = nn.Sequential(
            single_conv(6, 32),
            single_conv(32, 32)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(32, 64),
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up1 = up(128)
        self.conv3 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.up2 = up(64)
        self.conv4 = nn.Sequential(
            single_conv(32, 32),
            single_conv(32, 32)
        )

        self.outc = outconv(32, 3)

    def forward(self, x):
        inx = self.inc(x)       #takes 6 input channels and ouput 64 channel

        down1 = self.down1(inx)     #average pooling with filter size 2, stride 1
        conv1 = self.conv1(down1)   #takes 64 input channels and ouput 128 channel

        down2 = self.down2(conv1)   #average pooling with filter size 2, stride 1
        conv2 = self.conv2(down2)   #takes 128 input channels and ouput 256 channel

        up1 = self.up1(conv2, conv1)    #decovolution, 256 input channel, 128 output channel
        conv3 = self.conv3(up1)         #takes 128 input channels and ouput 128 channel

        up2 = self.up2(conv3, inx)      #decovolution, 128 input channel, 64 output channel
        conv4 = self.conv4(up2)         #takes 64 input channels and ouput 64 channel

        out = self.outc(conv4)          #takes 64 input channels and ouput 3 channel
        return out

#end model 0
##########################################################################################


##########################################################################################
# model 1 moderate complexity
#
class CBDnet_1(nn.Module):
    # please implement CBDnet here, this is just return the same size image
    def __init__(self):
        super(CBDnet_1, self).__init__()
        #input image is 3 * 256 * 256
        self.fcn = FCN_1()            #CNN_E: takes an noisy observtion y and output esitmate the noise level map
        self.unet = UNet_1()          #UNet: performs image denoise

        #daniel#self.conv1 = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        noise_level = self.fcn(x)           #get the noise level map of input x
        concat_img = torch.cat([x, noise_level], dim=1)         #combine the two tensor together as an inout to unet
        out = self.unet(concat_img) + x     #taking both noisy image and noise level map as input is helpful in generalizing the learned model to images beyond the noise model
        return out
        #daniel#x = F.relu(self.conv1(x)) # (256+2*1-3)/1+1=256
        #return x

#FCN class estimate noise level of the input signal
class FCN_1(nn.Module):  # CNN_E
    def __init__(self):
        super(FCN_1, self).__init__()
        self.fcn = nn.Sequential(           #5 CNN with channel size all set to 32
            nn.Conv2d(3, 32, 3, padding=1),     #conv2d ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fcn(x)

#UNet perform denoise through three levels
class UNet_1(nn.Module):
    def __init__(self):
        super(UNet_1, self).__init__()

        self.inc = nn.Sequential(
            single_conv(6, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 3)

    def forward(self, x):
        inx = self.inc(x)       #takes 6 input channels and ouput 64 channel

        down1 = self.down1(inx)     #average pooling with filter size 2, stride 1
        conv1 = self.conv1(down1)   #takes 64 input channels and ouput 128 channel

        down2 = self.down2(conv1)   #average pooling with filter size 2, stride 1
        conv2 = self.conv2(down2)   #takes 128 input channels and ouput 256 channel

        up1 = self.up1(conv2, conv1)    #decovolution, 256 input channel, 128 output channel
        conv3 = self.conv3(up1)         #takes 128 input channels and ouput 128 channel

        up2 = self.up2(conv3, inx)      #decovolution, 128 input channel, 64 output channel
        conv4 = self.conv4(up2)         #takes 64 input channels and ouput 64 channel

        out = self.outc(conv4)          #takes 64 input channels and ouput 3 channel
        return out

#end model 1
##########################################################################################


##########################################################################################
#model 2 high complexity
class CBDnet_2(nn.Module):
    # please implement CBDnet here, this is just return the same size image
    def __init__(self):
        super(CBDnet_2, self).__init__()
        # input image is 3 * 256 * 256
        self.fcn = FCN_2()  # CNN_E: takes an noisy observtion y and output esitmate the noise level map
        self.unet = UNet_2()  # UNet: performs image denoise

        # daniel#self.conv1 = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        noise_level = self.fcn(x)  # get the noise level map of input x
        concat_img = torch.cat([x, noise_level], dim=1)  # combine the two tensor together as an inout to unet
        out = self.unet(
            concat_img) + x  # taking both noisy image and noise level map as input is helpful in generalizing the learned model to images beyond the noise model
        return out
        # daniel#x = F.relu(self.conv1(x)) # (256+2*1-3)/1+1=256
        # return x


# FCN class estimate noise level of the input signal
class FCN_2(nn.Module):  # CNN_E
    def __init__(self):
        super(FCN_2, self).__init__()
        self.fcn = nn.Sequential(  # 5 CNN with channel size all set to 32
            nn.Conv2d(3, 32, 3, padding=1),
            # conv2d ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fcn(x)


# UNet perform denoise through three levels
class UNet_2(nn.Module):
    def __init__(self):
        super(UNet_2, self).__init__()

        self.inc = nn.Sequential(
            single_conv(6, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 3)

    def forward(self, x):
        inx = self.inc(x)  # takes 6 input channels and ouput 64 channel

        down1 = self.down1(inx)  # average pooling with filter size 2, stride 1
        conv1 = self.conv1(down1)  # takes 64 input channels and ouput 128 channel

        down2 = self.down2(conv1)  # average pooling with filter size 2, stride 1
        conv2 = self.conv2(down2)  # takes 128 input channels and ouput 256 channel

        up1 = self.up1(conv2, conv1)  # decovolution, 256 input channel, 128 output channel
        conv3 = self.conv3(up1)  # takes 128 input channels and ouput 128 channel

        up2 = self.up2(conv3, inx)  # decovolution, 128 input channel, 64 output channel
        conv4 = self.conv4(up2)  # takes 64 input channels and ouput 64 channel

        out = self.outc(conv4)  # takes 64 input channels and ouput 3 channel
        return out


##########################################################################################
# Below are the hpler function for UNet
# this class implements a signal conv2d layer with relu activation
class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# this class doing the transpose convolution (can be view as deconvolution) visiualizetion of the opration can be
# found here: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
# ConvTranspose2d ref: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW, size the output
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x


# final unet ouput layer
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

#end model 2
##########################################################################################


# the training code is adapted from tut 3a, adding data normalization and weight decay to prevent overfitting
def train(model, batch_size=20, num_epochs=1, learning_rate=0.01, train_type=0, weight_decay=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0  # the number of iterations
    start_time = time.time()
    for epoch in range(num_epochs):
        mini_b = 0
        for imgs, labels in iter(train_loader):
            imgs, labels = utility.normalization(imgs, labels)

            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################

            # update
            out = model(imgs)  # forward pass

            # print(imgs.shape)
            # print(labels.shape)

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

    path = "model_parameter"
    torch.save(model.state_dict(),
               path + "lr_{} batch_size_{} weight_decay{}".format(learning_rate, batch_size, weight_decay))

    end_time = time.time()

    #utility.loss_plotting(iters, losses)

    print("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % (
        (end_time - start_time), ((end_time - start_time) / num_epochs)))


if __name__ == '__main__':
    use_cuda = True
    num_workers = 0
    weight_decay = 0.001
    num_epochs = 20
    learning_rate = 7e-5
    batch_size =41

    delta_batch_size = 1
    delta_learning_rate = 0.5e-5
    delta_weight_decay = 0.0001

    best_batch_size = 0
    best_learning_rate = -1.
    best_weight_decay = -1.

    count_hype = 0
    iteration = 5
    psnr_prev = 35.5



    while (True):
        for _ in range(iteration):
            train_loader, val_loader, test_loader = get_dataloaders(train_path="../Dataset/Merged_Dataset/train",
                                                                    val_path="../Dataset/Merged_Dataset/val",
                                                                    test_path="../Dataset/Merged_Dataset/test",
                                                                    batch_size=batch_size)

            # proper model
            train_model = True  # if we need to train the model

            model = utility.load_model(train_model, learning_rate, batch_size, weight_decay)

            if use_cuda and torch.cuda.is_available():
                model.cuda()
                print('CUDA is available!  Training on GPU ...')
            else:
                print('CUDA is not available.  Training on CPU ...')

            if (train_model):
                train(model, batch_size=batch_size, num_epochs=num_epochs, weight_decay=weight_decay,
                      learning_rate=learning_rate)

            count = utility.save_model_output(model, use_cuda)

            psnr_new = utility.PSNR(model, count, psnr_predict_clean=True)
            ssim_new = utility.SSIM(model, count, psnr_predict_clean=True)

            print("The Average PSNR between model prediction and clean is {}".format(psnr_new))
            print("The Average SSIM between model prediction and clean is {}".format(ssim_new
                ))

            print("The Average PSNR has improved by {}".format(
                utility.PSNR(model, count, psnr_predict_clean=True) - utility.PSNR(model, count,
                                                                                   psnr_predict_clean=False)))
            print("The Average SSIM has improved by {}".format(
                utility.SSIM(model, count, psnr_predict_clean=True) - utility.SSIM(model, count,
                                                                                   psnr_predict_clean=False)))

            if (psnr_new > psnr_prev):
                best_batch_size = batch_size
                best_learning_rate = learning_rate
                best_weight_decay = weight_decay

                with open("log.txt",'a') as fd:
                    print("writing to file")
                    fd.write("The current best hyperparameters are: {} {} {}\n".format(best_batch_size,best_learning_rate,best_weight_decay))
                    fd.write("PSNR is {}, SSIM is {}\n".format(psnr_new, ssim_new))
                if (count_hype % 3 == 0):
                    batch_size -= delta_batch_size
                elif (count_hype % 3 == 1):
                    learning_rate -= delta_learning_rate
                else:
                    weight_decay -= delta_weight_decay
                psnr_prev = psnr_new
            else:
                if (count_hype % 3 == 0 and batch_size <= 75):
                    batch_size += delta_batch_size
                    #if (batch_size )
                elif (count_hype % 3 == 1):
                    learning_rate += delta_learning_rate
                else:
                    weight_decay += delta_weight_decay


            count_hype += 1