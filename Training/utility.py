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
    min_imgs = torch.min(imgs).cpu().detach().numpy().tolist()
    max_imgs = torch.max(imgs).cpu().detach().numpy().tolist()

    min_labels = torch.min(labels).cpu().detach().numpy().tolist()
    max_labels = torch.max(labels).cpu().detach().numpy().tolist()

    return (imgs-min_imgs) / (max_imgs - min_imgs),(labels - min_labels) / (max_labels - min_labels),min_imgs,max_imgs,min_labels,max_labels


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
    model = CNN.CBDnet_1()

    if os.path.exists(path) and train_model == False:
        print("loading exiting model")
        state = torch.load(path)
        model.load_state_dict(state)

    return model

def tensor_to_img(tensor,min_labels,max_labels):
    tensor = tensor.cpu().detach().numpy().squeeze(axis=0)
    tensor = np.transpose(tensor, [1, 2, 0])

    tensor = tensor * (max_labels - min_labels) + min_labels
    tensor = (tensor).astype(np.uint8)
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
        imgs, labels,min_imgs,max_imgs,min_labels,max_labels = normalization(imgs, labels)

        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################

        # from tensor to numpy img
        _,out = model(imgs)


        imgs = tensor_to_img(imgs,min_imgs,max_imgs)
        out = tensor_to_img(out,min_imgs,max_imgs)
        labels = tensor_to_img(labels,min_labels,max_labels)

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
information of how to use library PSNR can be found in here
https://dsp.stackexchange.com/questions/38065/peak-signal-to-noise-ratio-psnr-in-python-for-an-image
IEEE citation:
Sudip DasSudip Das                    19111 gold badge11 silver badge44 bronze badges, Amir KhakpourAmir Khakpour                    9111 silver badge11 bronze badge, Himanshu TyagiHimanshu Tyagi                    8111 silver badge22 bronze badges, Shuai YanShuai Yan                    5111 silver badge11 bronze badge, and Dan BoschenDan Boschen                    33.7k22 gold badges3434 silver badges9393 bronze badges, “Peak signal to noise ratio (PSNR) in python for an image,” Signal Processing Stack Exchange, 01-Dec-1964. [Online]. Available: https://dsp.stackexchange.com/questions/38065/peak-signal-to-noise-ratio-psnr-in-python-for-an-image. [Accessed: 12-Apr-2022]. 
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
ssim_predict_clean = true: its the SSIM between the prediction of our model and clean img
ssim_predict_clean = false: its the SSIM between the dirty and clean img
information of how to use library call to calcluate SSIM can be found here
https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
IEEE citation:
Linus, A. Rosebrock, Kushi, Giri, Mourad, L. Loja, R. Patel, J. Cohen, S. Barnes, Simon, Pavan, Andreas, Bill, Vijeta, Marc, Harrison, Pranali, y0c0, S. Leach, LianMing, Andrew, P. Srinivasan, Anwar, Yitzhak, Ambika, Erika, Manu, Vin, Pochao, P. K, M. Prasad, Ilja, Bilal, Anh, Gandhirajan, A. Bhalla, Vinay, R. V.K, Ravikumar, Nihel, Ju, Esteban, Sudheendra, S. R, Ali.K, Nut, Alex, Coby, S. Kunkel, David, D. agarwal, Viktor, IndhraG, Manbodh, heetak Chung, Parisa, Parisa, K. Mukherjee, D. Mike, Midun, Quinn, Alex, Kiran, I. Ahmad, Rohit, Ashley, S. Bhansali, N. Pham, D. Suprianto, S. H, J. Johnsen, S. Sturges, Zheng, Yaswanth, Ali, Sholi, Ash, Anirudh, M. Jha, M. Maheshwari, N. Jain, Shahane, Mracv, C. Greco, Andrei, Tomek, B. Agarwal, Pavan, Ravi, S. V. S, Chin, Raj, J. J, Mario, Frank, Denis, Barry, S. Kotgire, and Javier, “Image difference with opencv and python,” PyImageSearch, 07-Jul-2021. [Online]. Available: https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/. [Accessed: 12-Apr-2022]. 
'''
def SSIM(model, count, psnr_predict_clean):
    path = "./output/"
    ssim = []

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
