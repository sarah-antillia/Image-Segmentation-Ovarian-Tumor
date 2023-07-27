# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# split_master.py
# 2023/07/22

"""
./Ovarin-Tumor-master
 ├─images
 └─masks

"""
# into test, train and valid dataset.

"""
Ovarin-Tumor
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
  ├─images
  └─masks
"""


import os
import sys
import glob
import shutil

import traceback
import random

def split_master(images_dir, masks_dir, output_dir):
  image_files = glob.glob(images_dir + "/*.JPG")
  random.shuffle(image_files)
  num = len(image_files)
  num_train = int(num * 0.8)
  num_valid = int(num * 0.15)
  num_test  = int(num * 0.05)
  print("num_train {}".format(num_train))
  print("num_valid {}".format(num_valid))
  print("num_test  {}".format(num_test ))

  train_files = image_files[:num_train]
  valid_files = image_files[num_train:num_train+ num_valid]
  test_files  = image_files[num_train+ num_valid:]
  train_dir   = os.path.join(output_dir, "train")
  valid_dir   = os.path.join(output_dir, "valid")
  test_dir    = os.path.join(output_dir, "test")
  copy(train_files, masks_dir, train_dir)
  copy(valid_files, masks_dir, valid_dir)
  copy(test_files,  masks_dir, test_dir )


def copy(image_files, masks_dir, dataset_dir):
  out_images_dir = os.path.join(dataset_dir, "images")
  out_masks_dir  = os.path.join(dataset_dir, "masks")

  if not os.path.exists(out_images_dir):
    os.makedirs(out_images_dir)
  if not os.path.exists(out_masks_dir):
    os.makedirs(out_masks_dir)

  for image_file in image_files:
    shutil.copy2(image_file, out_images_dir)
    print("Copied {} to {}".format(image_file, out_images_dir))

    basename = os.path.basename(image_file)
    mask_filepath = os.path.join(masks_dir, basename)
    shutil.copy2(mask_filepath, out_masks_dir)
    print("Copied {} to {}".format(mask_filepath, out_masks_dir))


if __name__ == "__main__":
  try:
    images_dir = "./Ovarian-Tumor-master/images/"
    masks_dir  = "./Ovarian-Tumor-master/masks/"
    output_dir = "./Ovarian-Tumor/"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    split_master(images_dir, masks_dir, output_dir)

  except:
    traceback.print_exc()

