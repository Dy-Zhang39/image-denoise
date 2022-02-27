from skimage.transform import rescale, resize, downscale_local_mean
from skimage.util import random_noise
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, abspath, isfile, join, isdir
from os import listdir, rmdir, mkdir
import re
import shutil
import sys
from PIL import Image


if ('-h' in sys.argv or '--help' in sys.argv):
	print("You must run this script in the dataset folder. For example, if the dataset is PolyU, ")
	print("run this script in Dataset/PolyU folder.")
	print("Usage: python process_images.py <DATASET> [-f]")
	print("<DATASET>: one of NIND, PolyU, SSID")
	print("Option: -regen_all: exclusive of -regen_patch. Force regenerate all image patches even if they already exist on disk")
	print("Option -regen_patch: exclusive of -regen_all. Assumes that images are already properly-named. Only regenerates image patches from images.")
	quit()

# Argument -regen_all:
if ('-regen_all' in sys.argv):
	force_generate_all = True
	print("Will rename all images and force regenerate all image patches")
else:
	force_generate_all = False

# Argument -regen_patch:
if ('-regen_patch' in sys.argv):
	force_generate_patch = True
	print("Will only create image patches from already-renamed images")
else:
	force_generate_patch = False


if ("NIND" in sys.argv):
	current_dataset = "NIND"
elif "PolyU" in sys.argv:
	current_dataset = "PolyU"
elif "SSID" in sys.argv:
	current_dataset = "SSID"
elif "Custom" in sys.argv:
	current_dataset = "Custom"
else:
	print("Error: you did not specify the correct dataset")
	quit()

# IMPORTANT: Run this script in the dataset folder. For example, if the dataset is PolyU,
# run this script in Dataset/PolyU
renamed_image_files = []
renamed_image_dir = join(current_dataset, "Renamed_Images")

# If Renamed_Images does not exist, create it
if (not isdir(renamed_image_dir)):
	mkdir(renamed_image_dir)
elif len(listdir(renamed_image_dir)) != 0 and force_generate_all:
	shutil.rmtree(renamed_image_dir)
	mkdir(renamed_image_dir)



# We hope to identify clean/dirty images with each dataset. The naming convention
# will differ between datasets. The following code takes whatever images there are,
# and puts them into the same folder called "Renamed_Images" with consistent naming conventions
if (force_generate_patch == False):
	if current_dataset == "PolyU" and len(listdir(renamed_image_dir)) == 0:
		source_dir = join(current_dataset, "Original_Images_Untouched")
		source_image_files = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
		assert(len(source_image_files) % 2 == 0)

		# iterate over each pair of files (assuming that dirty/clean images are adjacent to each other)
		# and rename them as appropriate
		dirty_match_re = re.compile('(.+)_Real.JPG')
		clean_match_re = re.compile('(.+)_mean.JPG')

		for idx in range(0, len(source_image_files), 2):
			clean_image_file = source_image_files[idx]
			dirty_image_file = source_image_files[idx+1]

			dirty_match = dirty_match_re.search(dirty_image_file)
			clean_match = clean_match_re.search(clean_image_file)

			# group 1 captures each image's prefix. This is a sanity check that ensures
			# that image names in the dataset are consistent.
			assert(dirty_match.group(1) == clean_match.group(1))

			new_dirty_name = "PolyU_" + dirty_match.group(1) + "_dirty.jpg"
			new_clean_name = "PolyU_" + dirty_match.group(1) + "_clean.jpg"
			shutil.copy(join(source_dir, dirty_image_file), join(renamed_image_dir, new_dirty_name))
			shutil.copy(join(source_dir, clean_image_file), join(renamed_image_dir, new_clean_name))
			renamed_image_files.append(new_dirty_name)
			renamed_image_files.append(new_clean_name)

	elif current_dataset == "SSID" and len(listdir(renamed_image_dir)) == 0:
		source_dir = join(current_dataset, "Original_Images_Untouched")

		# within SSID dataset, each scene is divided into four images - two clean, two dirty.
		# We will pick only one of the two image pairs, labelled as 010.
		# Another complexity is that we have to save the image in jpg instead of png.
		scene_dirs = [d for d in listdir(source_dir) if (d != "." and d != ".." and isdir(join(source_dir, d)))]

		dirty_match_re = re.compile("([0-9]+)_NOISY_SRGB_010.PNG")
		clean_match_re = re.compile("([0-9]+)_GT_SRGB_010.PNG")

		for d in scene_dirs:
			image_dir = join(source_dir, d)

			entries = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
			assert(len(entries) == 4)

			found_dirty = False
			found_clean = False
			for entry in entries:

				if found_dirty == False:
					dirty_match = dirty_match_re.search(entry)
					if (dirty_match):
						new_dirty_name = "SSID_" + dirty_match.group(1) + "_dirty.jpg"
						found_dirty = True
						dirty_image = Image.open(join(image_dir, entry)).convert('RGB')
						dirty_image.save(join(renamed_image_dir, new_dirty_name))
						renamed_image_files.append(new_dirty_name)

				if found_clean == False:
					clean_match = clean_match_re.search(entry)
					if (clean_match):
						new_clean_name = "SSID_" + clean_match.group(1) + "_clean.jpg"
						found_clean = True
						clean_image = Image.open(join(image_dir, entry)).convert('RGB')
						clean_image.save(join(renamed_image_dir, new_clean_name))
						renamed_image_files.append(new_clean_name)

			assert(found_dirty and found_clean)
	elif current_dataset == "NIND" and len(listdir(renamed_image_dir)) == 0:
		source_dir = join(current_dataset, "Original_Images_Untouched")

		# for NIND dataset, many images have not been compressed and are massive. The processing part
		# involves ensuring that all images are renamed, compressed, and saved as JPG.
		# Similar to SSID, each scene is stored in its own folder. However, unlike SSID, rather than clean
		# and dirty images, this dataset comes with a range of images with different ISO, lower ISO = cleaner.
		# We pick output a low-ISO image and a high-ISO image from each set.

		scene_dirs = [d for d in listdir(source_dir) if (d != "." and d != ".." and isdir(join(source_dir, d)))]
		iso_match_re = re.compile("NIND_(.*)_ISO([0-9]*)\.(jpg|JPG|png|PNG)")

		for d in scene_dirs:
			image_dir = join(source_dir, d)
			entries = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

			smallest_iso = 9898989
			largest_iso = -1
			for entry in entries:
				iso_match = iso_match_re.search(entry)
				if iso_match:
					scene_name = iso_match.group(1)
					pic_format = iso_match.group(3)
					current_iso = int(iso_match.group(2))

					smallest_iso = min(smallest_iso, current_iso)
					largest_iso = max(largest_iso, current_iso)

			if (smallest_iso != 9898989 and smallest_iso != largest_iso and largest_iso != -1):
				# this means that at least 2 pictures with unique ISO's have been found.
				# Use these images for training.
				clean_entry = "NIND_" + scene_name + "_ISO" + str(smallest_iso) + "." + pic_format
				dirty_entry = "NIND_" + scene_name + "_ISO" + str(largest_iso) + "." + pic_format

				new_clean_name = "NIND_" + scene_name + "_clean.jpg"
				new_dirty_name = "NIND_" + scene_name + "_dirty.jpg"

				requires_conversion = pic_format == "PNG" or pic_format == "png"

				clean_image = Image.open(join(image_dir, clean_entry))
				dirty_image = Image.open(join(image_dir, dirty_entry))

				if (requires_conversion):
					clean_image = clean_image.convert('RGB')
					dirty_image = dirty_image.convert('RGB')
				clean_image.save(join(renamed_image_dir, new_clean_name))
				dirty_image.save(join(renamed_image_dir, new_dirty_name))

				renamed_image_files.append(new_clean_name)
				renamed_image_files.append(new_dirty_name)


	elif current_dataset == "Custom" and len(listdir(renamed_image_dir)) == 0:
		source_dir = join(current_dataset, "Original_Images")

		# This dataset only contains original "clean" images. The goal is to
		# add artificial Gaussian noise to produce "clean/dirty" pairs
		source_image_files = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]

		image_idx = 0
		for img in source_image_files:
			src_image_path = join(source_dir, img)
			clean_image = plt.imread(src_image_path)
			dirty_image = random_noise(clean_image)

			new_clean_name = "Custom_{}_clean.jpg".format(image_idx)
			new_dirty_name = "Custom_{}_dirty.jpg".format(image_idx)
			renamed_image_files.append(new_dirty_name)
			renamed_image_files.append(new_clean_name)

			plt.imsave(join(renamed_image_dir, new_clean_name), clean_image)
			plt.imsave(join(renamed_image_dir, new_dirty_name), dirty_image)
			image_idx += 1


# This part should be generalized. It assumes that all images are in a folder called Renamed_Images.
# It takes in each image, resizes it to 1024 x 1024, and generates 16 patches of 256 x 256 for each image.
# It then saves these images in the Patches folder

# if force_generate_patch is true, the above loops is skipped and we assume that all images are in /Renamed_Images
if (force_generate_patch == True):
	renamed_image_files = [f for f in listdir(renamed_image_dir) if isfile(join(renamed_image_dir, f))]
	assert(len(renamed_image_files) != 0)


patch_count = 0
if (len(renamed_image_files) > 0):

	patches_dir = join(current_dataset, "Patches")
	if (not isdir(patches_dir)):
		mkdir(patches_dir)
	elif len(listdir(patches_dir)) != 0 and (force_generate_all or force_generate_patch):
		shutil.rmtree(patches_dir)
		mkdir(patches_dir)


	for image in renamed_image_files:
		src_image_path = join(renamed_image_dir, image)

		src_image = plt.imread(src_image_path)
		image_resized = resize(src_image, (1024, 1024), anti_aliasing = True)

		# generate 16 patches
		for r in range(0, 4):
			for c in range(0, 4):
				patch_count += 1
				patch = image_resized[r * 256 : (r+1) * 256, c * 256 : (c+1) * 256]

				# Note: splitting on "_" will put dirty and clean patches side by side.
				#       splitting on "." will put all the clean patches for an image together.
				split_idx = image.rfind("_")
				patch_name = image[:split_idx] + "{}{}".format(r, c) + image[split_idx:]

				plt.imsave(join(patches_dir, patch_name), patch)

	print("Success: generated {} patches in directory {}".format(patch_count, patches_dir))
