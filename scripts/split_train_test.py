#!/usr/bin/env python3
import sys
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

directory = sys.argv[1]
train_dir = os.path.join(directory, "00train/")
test_dir = os.path.join(directory, "00test/")

imagelist = []
for filename in Path(directory).rglob("*.png"):
    imagelist.append(filename)
for filename in Path(directory).rglob("*.jpg"):
    imagelist.append(filename)
print(f"Found {len(imagelist)} images in {directory}")

train, test = train_test_split(imagelist, test_size=0.15, shuffle=True)

for img in train:
    shutil.copy(img, train_dir)
for img in test:
    shutil.copy(img, test_dir)