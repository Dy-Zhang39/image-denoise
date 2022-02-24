import os
import shutil
import numpy as np

'''
This script generates the training, validation, and testing datasets by randomly
choosing image patches from existing datasets and dividing them into the sets
in 80-10-10 split.
We will have 1920 patches in total, so 1536 in training, 192 in val/testing each.
The 1920 patches is because the smallest dataset, PolyU, contains 40 pairs of clean/dirty
images, which produces 640 patches. In order to not let it be underrepresented,
we will pick 640 patches from the other 2 training sets too.

Output folder structure:
train
 dirty
  -img1_dirty.jpg
 clean
  -img1_clean.jpg
val
 dirty
 clean
test
 dirty
 clean

Another consideration is to have patches from images never seen before in validation
and testing. Hence, for training data we will randomly pick 1536 images from the first 90%
of the data; the unpicked images will be merged with 5% of the data to choose validation
set from; then, pick test data from the remaining images.

This script assumes a consistent layout in each folder containing image patches.
Especially, clean patches are in odd indices, and dirty patches are right after it
in even indices. This layout can be maintained by process_images.py
'''
np.random.seed(69)
datasets_to_process = ["NIND", "PolyU", "SSID"]
dest_dataset = "Merged_Dataset"

# Create all folders if necessary
if not os.path.isdir(dest_dataset):
    os.mkdir(dest_dataset)
for x in ['train', 'val', 'test']:
    x_path = os.path.join(dest_dataset, x)
    if not os.path.isdir(x_path):
        os.mkdir(x_path)
    for y in ['clean', 'dirty']:
        y_path = os.path.join(x_path, y)
        if not os.path.isdir(y_path):
            os.mkdir(y_path)


ranges = [('train', 0, 0.90, 512), ('val', 0.90, 0.95, 64), ('test', 0.95, 1, 64)]

for dataset in datasets_to_process:
    patch_path = os.path.join(dataset, "Patches")
    all_clean_data = [f for f in os.listdir(patch_path) if "clean" in f]
    num_clean_data = len(all_clean_data)
    carry_over_data = []

    for set_and_range in ranges:
        set = set_and_range[0]

        dest_clean_folder = os.path.join(os.path.join(dest_dataset, set), 'clean')
        dest_dirty_folder = os.path.join(os.path.join(dest_dataset, set), 'dirty')

        # to avoid diluting, only allow carrying over the set_and_range[3] amount of images.
        # This should be just enough for PolyU, and adequate for the rest
        data_to_pick_from =  carry_over_data[0:set_and_range[3]]
        data_to_pick_from += all_clean_data[int(num_clean_data * set_and_range[1]) : int(num_clean_data * set_and_range[2])]
        np.random.shuffle(data_to_pick_from)

        print("Working on dataset ", set)
        for i in range(0, set_and_range[3]):
            clean_image_name = data_to_pick_from[i]
            dirty_image_name = clean_image_name[0:clean_image_name.find('clean')] + 'dirty.jpg'
            clean_image_path = os.path.join(patch_path, clean_image_name)
            dirty_image_path = os.path.join(patch_path, dirty_image_name)

            assert(os.path.isfile(clean_image_path) and os.path.isfile(dirty_image_path))

            shutil.copy(clean_image_path, os.path.join(dest_clean_folder, clean_image_name))
            shutil.copy(dirty_image_path, os.path.join(dest_dirty_folder, dirty_image_name))

        carry_over_data = data_to_pick_from[set_and_range[3]:]
