import os
import torch
import numpy as np
import CNN
import cv2
from PIL import Image
import matplotlib.pyplot as plt # for plotting
from dataloader import get_dataloaders
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
def load_model(train_model, learning_rate, batch_size, weight_decay):
    path = "model_parameter"+ "lr_{} batch_size_{} weight_decay{}".format(learning_rate, batch_size, weight_decay)
    model = CNN.CBDnet()

    if os.path.exists(path) and train_model == False:
        print("loading exiting model")
        state = torch.load(path)
        model.load_state_dict(state)

    return model

def tensor_to_img(tensor):
    tensor = tensor.cpu().detach().numpy().squeeze(axis=0)
    tensor = np.transpose(tensor, [1, 2, 0])
    tensor = (tensor * 255).astype(np.uint8)
    tensor = np.clip(tensor, 0, 255)

    return Image.fromarray(tensor)

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


        imgs = tensor_to_img(imgs)
        out = tensor_to_img(out)
        labels = tensor_to_img(labels)

        #dir creation
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(os.path.join(path, "out")):
            os.mkdir(os.path.join(path, "out"))
        if not os.path.exists(os.path.join(path, "clean")):
            os.mkdir(os.path.join(path, "clean"))
        if not os.path.exists(os.path.join(path, "dirty")):
            os.mkdir(os.path.join(path, "dirty"))

        # numpy img to actual img
        imgs.save("{}dirty/test{}.jpg".format(path, count))
        out.save("{}out/test{}.jpg".format(path, count))
        labels.save("{}clean/test{}.jpg".format(path, count))

        count+=1

    return count

'''
Calculate average PSNR among all testset data
psnr_predict_clean = true: its the PSNR between the prediction of our model and clean img
psnr_predict_clean = false: its the PSNR between the dirty and clean img
'''
def PSNR(model, count, psnr_predict_clean):
    path = "./output/"
    psnr = []

    for i in range(count):
        if (psnr_predict_clean):
            img1 = cv2.imread("{}out/test{}.jpg".format(path,i))
        else:
            img1 = cv2.imread("{}dirty/test{}.jpg".format(path,i))
        img2 = cv2.imread("{}clean/test{}.jpg".format(path, i))
        psnr.append(cv2.PSNR(img1, img2))

    return sum(psnr)/len(psnr)

'''
Calculate average SSIM among all testset data
psnr_predict_clean = true: its the PSNR between the prediction of our model and clean img
psnr_predict_clean = false: its the PSNR between the dirty and clean img
'''
def SSIM(model, count, psnr_predict_clean):
    path = "./output/"
    ssim = []
    # https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
    for i in range(count):
        if (psnr_predict_clean):
            img1 = cv2.imread("{}out/test{}.jpg".format(path, i))
        else:
            img1 = cv2.imread("{}dirty/test{}.jpg".format(path, i))
        img2 = cv2.imread("{}clean/test{}.jpg".format(path, i))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        score,_ = compare_ssim(gray1, gray2, full=True)
        ssim.append(score)

    return sum(ssim)/len(ssim)
