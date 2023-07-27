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

# ImageMaskDataset.py
# 2023/05/31 to-arai Modified to use config_file

import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
import traceback
from ConfigParser import ConfigParser

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"
MASK   = "mask"

class ImageMaskDataset:

  def __init__(self, config_file):
    config = ConfigParser(config_file)
    self.image_width    = config.get(MODEL, "image_width")
    self.image_height   = config.get(MODEL, "image_height")
    self.image_channels = config.get(MODEL, "image_channels")
    self.train_dataset  = [ config.get(TRAIN, "image_datapath"),
                            config.get(TRAIN, "mask_datapath")]
    
    self.eval_dataset   = [ config.get(EVAL, "image_datapath"),
                            config.get(EVAL, "mask_datapath")]

    self.binarize  = config.get(MASK, "binarize")
    self.threshold = config.get(MASK, "threshold")
    self.blur_mask = config.get(MASK, "blur")

    #Fixed blur_size
    self.blur_size = (3, 3)


  # If needed, please override this method in a subclass derived from this class.
  def create(self, dataset = TRAIN,  debug=False):
    if not dataset in [TRAIN, EVAL]:
      raise Exception("Invalid dataset")
    image_datapath = None
    mask_datapath  = None
  
    [image_datapath, mask_datapath] = self.train_dataset
    if dataset == EVAL:
      [image_datapath, mask_datapath] = self.eval_dataset

    image_files  = glob.glob(image_datapath + "/*.jpg")
    image_files += glob.glob(image_datapath + "/*.png")
    image_files += glob.glob(image_datapath + "/*.bmp")
    image_files += glob.glob(image_datapath + "/*.tif")
    image_files  = sorted(image_files)

    mask_files   = None
    if os.path.exists(mask_datapath):
      mask_files  = glob.glob(mask_datapath + "/*.jpg")
      mask_files += glob.glob(mask_datapath + "/*.png")
      mask_files += glob.glob(mask_datapath + "/*.bmp")
      mask_files += glob.glob(mask_datapath + "/*.tif")
      mask_files  = sorted(mask_files)
      
      if len(image_files) != len(mask_files):
        raise Exception("FATAL: Images and masks unmatched")
      
    num_images  = len(image_files)
    if num_images == 0:
      raise Exception("FATAL: Not found image files")
    
    X = np.zeros((num_images, self.image_height, self.image_width, self.image_channels), dtype=np.uint8)

    Y = np.zeros((num_images, self.image_height, self.image_width, 1                ), dtype=np.bool)

    for n, image_file in tqdm(enumerate(image_files), total=len(image_files)):
  
      image = cv2.imread(image_file)
      
      image = cv2.resize(image, dsize= (self.image_height, self.image_width), interpolation=cv2.INTER_NEAREST)
      X[n]  = image

      if mask_files != None:

        mask  = cv2.imread(mask_files[n])
        mask  = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask  = cv2.resize(mask, dsize= (self.image_height, self.image_width),   interpolation=cv2.INTER_NEAREST)

        # Binarize mask
        if self.binarize:
          mask[mask< self.threshold] =   0  
          mask[mask>=self.threshold] = 255

        # Blur mask 
        if self.blur_mask:
          mask = cv2.blur(mask, self.blur_size)
  
        mask  = np.expand_dims(mask, axis=-1)
        Y[n]  = mask

        if debug:
          cv2.imshow("---", mask)
          #plt.show()
          cv2.waitKey(27)
          input("XX")   
  
    return X, Y


    
if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"

    dataset = ImageMaskDataset(config_file)

    x_train, y_train = dataset.create(dataset=TRAIN, debug=False)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    x_test, y_test = dataset.create(dataset=EVAL)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

