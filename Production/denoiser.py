from tkinter import Tk
from tkinter.filedialog import askopenfilename
import sys
import math
# a hacky way to allow importing python files from other directories
sys.path.append("../Training")
sys.path.append("../Baseline")
import CNN
import autoencoder
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import utility
from skimage.transform import resize

PATCH_SIZE = 256

def tensor_img_to_np(tensor):
    tensor = tensor.cpu().detach().numpy().squeeze(axis=0)
    tensor = np.transpose(tensor, [1, 2, 0])
    tensor = np.clip(tensor, 0, 1)
    return tensor

use_cuda = torch.cuda.is_available() and True  # global override

Tk().withdraw()
filename = askopenfilename(filetypes=((("Images"), "*.jpg"), (("All files"), "*.*")))


# Todo: save the model in a better place after we obtain a satisfactory set of params
#model = autoencoder.Autoencoder_cnn()
#model.load_state_dict(torch.load("../Baseline/best parameter"))
model = CNN.CBDnet_1()
#model.load_state_dict(torch.load("../Training/model_parameter_epoch21_CBDnet_1_lr_0.0002 batch_size_8 weight_decay0.001_bestsum"))
model.load_state_dict(torch.load("../Training/model_parameter_epoch44_CBDnet_1_lr_0.0002 batch_size_8 weight_decay0.001_best_sum"))
if (use_cuda):
    model.cuda()


image_np = plt.imread(filename)
image_np = image_np / 255
#print(image_np.dtype)
#image_np = resize(image_np, (int(image_np.shape[0]/3), int(image_np.shape[1]/3)), anti_aliasing = True)
image_np = image_np.astype("float32")
print("Original image shape: ", image_np.shape)

# zero-pad the image so the height and width are the next multiple of 256
image_height = image_np.shape[0]
image_width = image_np.shape[1]

desired_width = PATCH_SIZE * math.ceil(image_width/PATCH_SIZE)
desired_height = PATCH_SIZE * math.ceil(image_height/PATCH_SIZE)
pad_left = int((desired_width - image_width) / 2)
pad_right = (desired_width - image_width) - pad_left
pad_top = int((desired_height - image_height) / 2)
pad_bottom = (desired_height - image_height) - pad_top

image_np = np.pad(image_np, pad_width=[(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)], mode='constant')

num_patch_width = int(desired_width / PATCH_SIZE)
num_patch_height = int(desired_height / PATCH_SIZE)
denoised_patches = np.zeros((num_patch_width, num_patch_height, PATCH_SIZE, PATCH_SIZE, 3))
transform = transforms.Compose(
        [transforms.ToTensor()])

total_patches = num_patch_width * num_patch_height
current_patch_idx = 0
# Divide the image into patches
for x in range(num_patch_width):
    for y in range(num_patch_height):
        current_patch_idx += 1
        patch = image_np[y * PATCH_SIZE : (y+1) * PATCH_SIZE,
                                 x * PATCH_SIZE : (x+1) * PATCH_SIZE, :]

        #plt.imshow(patch)
        #plt.show()

        patch = transform(patch)
        if (use_cuda):
            patch = patch.cuda()
        #patch = patch.view(PATCH_SIZE, PATCH_SIZE, 3)
        patch = torch.unsqueeze(patch, 0)
        patch = model(patch)

        # Pytorch's output is CHW, but we want to turn it into HWC for numpy
        denoised_patches[x, y] = tensor_img_to_np(patch)
        #plt.imshow(denoised_patches[x, y])
        #plt.show()
        print("Progress: {}%".format(100*(current_patch_idx / total_patches)))
        print(patch.shape)


# assemble all patches together
denoised_image = np.zeros((desired_height, desired_width, 3))
for x in range(num_patch_width):
    for y in range(num_patch_height):
        denoised_image[y * PATCH_SIZE : (y+1) * PATCH_SIZE,
                     x * PATCH_SIZE : (x+1) * PATCH_SIZE] = denoised_patches[x, y]


# crop out the added padding
if (pad_bottom == 0 and pad_right == 0):
    denoised_image = denoised_image[pad_top:, pad_left:, :]
elif (pad_bottom == 0):
    denoised_image = denoised_image[pad_top:, pad_left:-pad_right, :]
elif (pad_right == 0):
    denoised_image = denoised_image[pad_top:-pad_bottom, pad_left:, :]
else:
    denoised_image = denoised_image[pad_top:-pad_bottom, pad_left:-pad_right, :]

print("Denoised image shape: ", denoised_image.shape)
#plt.imshow(denoised_image)
#plt.show()

save_path = filename[:filename.rfind("/")] + "/result.jpg"
print("Done, result saved to {}".format(save_path))
plt.imsave(save_path, denoised_image)
