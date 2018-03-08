#/usr/bin/env python
'''
 For splitting data into training and test sets and any other Preprocessing
 Copies files from a input dir to configured training and testing locations. 
 Data is split as per config.
'''

import glob
import os
import shutil
import PIL
from random import shuffle
from PIL import Image

# Config
DATA_DIR='data'
INPUT_DIR='im1'
OUTPUT_DIR_TRAIN='train'
OUTPUT_DIR_TEST='test'
DATA_PATTERN = '*.png'
TRAINING_PERC = 0.7
SCALE_FACTOR = 0.5

# Processing functions
def process(dataset, root):
    for f in dataset:
		name = os.path.split(f)[-1]
		loc = os.path.join(root, name)
		img = Image.open(f)
		img = img.resize((int(img.size[0] * SCALE_FACTOR), int(img.size[1] * SCALE_FACTOR)), PIL.Image.ANTIALIAS)
		img.save(loc)


# Prepate dataset
in_dir = os.path.join(DATA_DIR, INPUT_DIR)
out_dirs = os.path.join(DATA_DIR, OUTPUT_DIR_TRAIN), os.path.join(DATA_DIR, OUTPUT_DIR_TEST)
dataset =  glob.glob(os.path.join(in_dir, DATA_PATTERN))
shuffle(dataset)

# Compute splits
num_files = len(dataset)
split_index = int(TRAINING_PERC * num_files)
train_set = dataset[:split_index]
test_set = dataset[split_index:]

# Create dirs if required
if not os.path.exists(out_dirs[0]):
	os.mkdir(out_dirs[0])
if not os.path.exists(out_dirs[1]):
	os.mkdir(out_dirs[1])

# Copy Files if no additional processing required
if not SCALE_FACTOR:
	for f in train_set:
		shutil.copy(f, out_dirs[0])
	for f in test_set:
		shutil.copy(f, out_dirs[1])
else:
#	process(train_set, out_dirs[0])
	process(test_set, out_dirs[1])
		