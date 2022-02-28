import os
import torch
import numpy as np
from Training import CNN
import cv2
from PIL import Image
import matplotlib.pyplot as plt # for plotting
from Training.dataloader import get_dataloaders
from skimage.metrics import structural_similarity as compare_ssim #https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python

# this file will contain utility functions

# scale the tensor in [0,1]
def normalization(imgs, labels):
    return imgs/255, labels/255


def loss_plotting(iters, losses):
    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

'''
if train_model = true, we will train our CBDnet model
else we will just used our previous trained model
'''
def load_model(train_model):
    path = "model parameter"
    model = CNN.CBDnet()

    if os.path.exists(path) and train_model == False:
        # load exiting model
        state = torch.load(path)
        model.load_state_dict(state)

    return model


'''
save the output of CBDnet model in jpg format under output directory
return the number of CBDnet output (testset number)
'''
def save_model_output(model, use_cuda):
    path = "./output/"
    _, _, test_loader = get_dataloaders(train_path="../Dataset/Merged_Dataset/train",
                                        val_path="../Dataset/Merged_Dataset/val",
                                        test_path="../Dataset/Merged_Dataset/test",
                                        batch_size=1)
    # k=1
    count = 0
    for imgs, labels in iter(test_loader):
        imgs, labels = normalization(imgs, labels)

        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################

        # from tensor to numpy img
        out = model(imgs)
        out = out.cpu().detach().numpy().squeeze(axis=0)
        labels = labels.cpu().detach().numpy().squeeze(axis=0)
        out = np.transpose(out, [1, 2, 0])
        labels = np.transpose(labels, [1, 2, 0])

        # denormalization
        out = (out * 255).astype(np.uint8)
        labels = (labels * 255).astype(np.uint8)

        out = np.clip(out, 0, 255)
        labels = np.clip(labels, 0, 255)

        #dir creation
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(os.path.join(path, "out")):
            os.mkdir(os.path.join(path, "out"))
        if not os.path.exists(os.path.join(path, "clean")):
            os.mkdir(os.path.join(path, "clean"))

        # numpy img to actual img
        out_image = Image.fromarray(out)
        out_image.save("{}out/test{}.jpg".format(path, count))
        clean_image = Image.fromarray(labels)
        clean_image.save("{}clean/test{}.jpg".format(path, count))

        count+=1

    return count

'''
Calculate average PSNR among all testset data
'''
def PSNR(model, count):
    path = "./output/"
    psnr = []

    for i in range(count):
        img1 = cv2.imread("{}out/test{}.jpg".format(path,i))
        img2 = cv2.imread("{}clean/test{}.jpg".format(path, i))
        psnr.append(cv2.PSNR(img1, img2))

    return sum(psnr)/len(psnr)

'''
Calculate average SSIM among all testset data
'''
def SSIM(model, count):
    path = "./output/"
    ssim = []
    # https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
    for i in range(count):
        img1 = cv2.imread("{}out/test{}.jpg".format(path, i))
        img2 = cv2.imread("{}clean/test{}.jpg".format(path, i))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        score,_ = compare_ssim(gray1, gray2, full=True)
        ssim.append(score)

    return sum(ssim)/len(ssim)