import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_image

# This script describes the Pytorch dataset subclass, as well as
# the dataloader class. It assumes that the folders are in the required
# format after running generate_dataset.py

class DENOISING_DATASET(torch.utils.data.Dataset):
    def __init__ (self, image_dir):
        self.clean_path = os.path.join(image_dir, "clean")
        self.dirty_path = os.path.join(image_dir, "dirty")
        self.clean_image_list = [f for f in os.listdir(self.clean_path) if
                        os.path.isfile(os.path.join(self.clean_path, f))]

    def __len__ (self):
        return len(self.clean_image_list)

    def __getitem__ (self, idx):
        clean_image_name = self.clean_image_list[idx]
        clean_image_path = os.path.join(self.clean_path, clean_image_name)
        dirty_image_name = clean_image_name[0:clean_image_name.find('clean')] + 'dirty.jpg'
        dirty_image_path = os.path.join(self.dirty_path, dirty_image_name)
        clean_image = read_image(clean_image_path)
        dirty_image = read_image(dirty_image_path)

        return dirty_image, clean_image

# inputs: the paths to training, validation and/or testing datasets.
# returns: the respective data loaders for the paths that are supplied
def get_dataloaders (batch_size = 32, train_path = None, val_path = None, test_path = None):
    train_loader = None
    test_loader = None
    val_loader = None

    if train_path:
        set = DENOISING_DATASET(train_path)
        train_loader = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=True)
    if val_path:
        set = DENOISING_DATASET(val_path)
        val_loader = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=True)
    if test_path:
        set = DENOISING_DATASET(test_path)
        test_loader = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader



if __name__ == '__main__':
    # visualize the data that we have
    train_loader, _, _ = get_dataloaders(train_path = "../Dataset/Merged_Dataset/train",
                                        batch_size = 1)
    k = 0
    for dirty, clean in train_loader:
        # since batch_size = 1, there is only 1 image in `images`
        # image is 3 * 256 * 256
        image = dirty[0]
        print(image.shape)
        # place the colour channel at the end, instead of at the beginning
        img = np.transpose(image, [1,2,0])
        plt.subplot(6, 2, k+1)
        plt.axis('off')
        plt.title('dirty')
        plt.imshow(img)

        image = clean[0]
        # place the colour channel at the end, instead of at the beginning
        img = np.transpose(image, [1,2,0])
        plt.subplot(6, 2, k+2)
        plt.axis('off')
        plt.title('clean')
        plt.imshow(img)

        k += 2
        if k > 10:
            break
    plt.show()
