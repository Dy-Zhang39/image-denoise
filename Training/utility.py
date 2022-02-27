import os
import torch
import numpy as np
import CNN
import cv2
from PIL import Image
import matplotlib.pyplot as plt # for plotting
from Training.dataloader import get_dataloaders

# this file will contain utility functions

# scale the tensor in [0,1]
def normalization(imgs, labels):
    return imgs/255, labels/255

def data_plotting(iters, losses):
    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    # plt.title("Training Curve")
    # plt.plot(iters, train_acc, label="Training")
    # plt.plot(iters, val_acc, label="Validation")
    # plt.xlabel("Iterations")
    # plt.ylabel("Validation Accuracy")
    # plt.legend(loc='best')
    # plt.show()

    # train_acc.append(get_accuracy(model, train=train_type))
    # print("Final Training Accuracy: {}".format(train_acc[-1]))
    # print("Final Validation Accuracy: {}".format(val_acc[-1]))

def load_model(train_model):
    path = "model parameter"
    model = CNN.CBDnet()

    if (os.path.exists(path) and train_model == False):
        # load exiting model
        state = torch.load(path)
        model.load_state_dict(state)

    return model

def PSNR(model):
    path = "./output/"
    _,_,test_loader = get_dataloaders(train_path="../Dataset/Merged_Dataset/train",
                                  val_path="../Dataset/Merged_Dataset/val",
                                  test_path="../Dataset/Merged_Dataset/test",
                                  batch_size=1)
    #k=1
    count = 0
    for imgs, labels in iter(test_loader):
        imgs, labels = normalization(imgs, labels)

        out = model(imgs)
        out = out.detach().numpy().squeeze(axis=0)
        labels = labels.detach().numpy().squeeze(axis=0)
        out = np.transpose(out, [1,2,0])
        labels = np.transpose(labels, [1, 2, 0])

        out = (out * 255).astype(np.uint8)
        labels = (labels * 255).astype(np.uint8)

        out = np.clip(out, 0, 255)
        labels = np.clip(labels, 0, 255)

        if (not os.path.exists(path)):
            os.mkdir(path)
        if (not os.path.exists(os.path.join(path, "out"))):
            os.mkdir(os.path.join(path, "out"))
        if (not os.path.exists(os.path.join(path,"clean"))):
            os.mkdir(os.path.join(path,"clean"))

        out_image = Image.fromarray(out)
        out_image.save("{}out/test{}.jpg".format(path,count))
        clean_image = Image.fromarray(labels)
        clean_image.save("{}clean/test{}.jpg".format(path, count))


        #plt.subplot(6, 2, k)
        #plt.imshow(out)

        #plt.subplot(6, 2, k + 1)
        #plt.imshow(labels)
        k+=2
        count+=1

        #if (k >= 10): break

    plt.show()

    psnr = []
    for i in range(count):
        img1 = cv2.imread("{}out/test{}.jpg".format(path,i))
        img2 = cv2.imread("{}clean/test{}.jpg".format(path, i))
        psnr.append(cv2.PSNR(img1, img2))

    return sum(psnr)/len(psnr)