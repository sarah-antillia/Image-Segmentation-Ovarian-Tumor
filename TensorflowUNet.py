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

# This is based on the code in the following web sites:

# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook

# 2. U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf

# You can customize your TensorflowUnNet model by using a configration file
# Example: train.config

# 2023/06/29 Updated create method to add BatchNormalization provied that 
#[model]
#normalization=True
# However, this True setting will not be recommended because this may have adverse effect
# on tiled_image_segmentation.

# 2023/07/01 Support Overlapped-Tiled-Image-Segmentation 
#[tiledinfer]
#overlapping=32
#Specify a pixel size to overlap-tiling.
#Specify 0 if you need no overlapping.

import os
import sys
import datetime

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import shutil

import sys
import glob
import traceback
import random
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image, ImageFilter, ImageOps
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D, Dropout, Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
from tensorflow.keras.losses import  BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
#from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ConfigParser import ConfigParser

from EpochChangeCallback import EpochChangeCallback
from GrayScaleImageWriter import GrayScaleImageWriter

from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss, dice_loss

"""
See: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
Module: tf.keras.metrics
Functions
"""

"""
See also: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py
"""

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"
# 2023/06/10
TILEDINFER = "tiledinfer"


BEST_MODEL_FILE = "best_model.h5"

class TensorflowUNet:

  def __init__(self, config_file):
    self.set_seed()
    self.config_file = config_file
    self.config    = ConfigParser(config_file)
    image_height   = self.config.get(MODEL, "image_height")
    image_width    = self.config.get(MODEL, "image_width")
    image_channels = self.config.get(MODEL, "image_channels")
    num_classes    = self.config.get(MODEL, "num_classes")
    base_filters   = self.config.get(MODEL, "base_filters")
    num_layers     = self.config.get(MODEL, "num_layers")
    self.model     = self.create(num_classes, image_height, image_width, image_channels, 
                            base_filters = base_filters, num_layers = num_layers)
    
    learning_rate  = self.config.get(MODEL, "learning_rate")
    clipvalue      = self.config.get(MODEL, "clipvalue", 0.2)

    self.optimizer = Adam(learning_rate = learning_rate, 
         beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, 
         clipvalue=clipvalue,  #2023/0626
         amsgrad=False)
    print("=== Optimizer Adam learning_rate {} clipvalue {}".format(learning_rate, clipvalue))
    
    self.model_loaded = False

    # 2023/05/20 Modified to read loss and metrics from train_eval_infer.config file.
    binary_crossentropy = tf.keras.metrics.binary_crossentropy
    binary_accuracy     = tf.keras.metrics.binary_accuracy

    # Default loss and metrics functions
    self.loss    = binary_crossentropy
    self.metrics = [binary_accuracy]
    
    # Read a loss function name from our config file, and eval it.
    # loss = "binary_crossentropy"
    self.loss  = eval(self.config.get(MODEL, "loss"))

    # Read a list of metrics function names, ant eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      self.metrics.append(eval(metric))
    
    print("--- loss    {}".format(self.loss))
    print("--- metrics {}".format(self.metrics))
    
    #self.model.trainable = self.trainable

    self.model.compile(optimizer = self.optimizer, loss= self.loss, metrics = self.metrics)
   
    show_summary = self.config.get(MODEL, "show_summary")
    if show_summary:
      self.model.summary()

  def set_seed(self, seed=137):
    print("=== set seed {}".format(seed))
    random.seed    = seed
    np.random.seed = seed
    tf.random.set_seed(seed)


  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = Input((image_height, image_width, image_channels))
    s= Lambda(lambda x: x / 255)(inputs)

    # normalization is False on default.
    normalization = self.config.get(MODEL, "normalization", dvalue=False)
    print("--- normalization {}".format(normalization))
    # Encoder
    dropout_rate = self.config.get(MODEL, "dropout_rate")
    enc         = []
    kernel_size = (3, 3)
    pool_size   = (2, 2)
    dilation    = (2, 2)
    strides     = (1, 1)
    # <experiment on="2023/06/20">
    # [model] 
    # Specify a tuple of base kernel size of odd number something like this: 
    # base_kernels = (5,5)
    base_kernels   = self.config.get(MODEL, "base_kernels", dvalue=(3,3))
    (k, k) = base_kernels
    kernel_sizes = []
    for n in range(num_layers):
      kernel_sizes += [(k, k)]
      k -= 2
      if k <3:
        k = 3
    rkernel_sizes =  kernel_sizes[::-1]
    rkernel_sizes = rkernel_sizes[1:] 
    # kernel_sizes will become a list [(7,7),(5,5), (3,3),(3,3)...] if base_kernels were (7,7)
    print("--- kernel_size   {}".format(kernel_sizes))
    print("--- rkernel_size  {}".format(rkernel_sizes))
    # </experiment>
    dilation = None
    try:
      dilation_ = self.config.get(MODEL, "dilation", (1, 1))
      (d1, d2) = dilation_
      if d1 == d2:
        dilation = dilation_
    except:
      pass

    dilations = []
    (d, d) = dilation
    for n in range(num_layers):
      dilations += [(d, d)]
      d -= 1
      if d <1:
        d = 1
    rdilations = dilations[::-1]
    rdilations = rdilations[1:]

    print("=== dilations  {}".format(dilations))
    print("=== rdilations {}".format(rdilations))

    for i in range(num_layers):
      filters = base_filters * (2**i)
      kernel_size = kernel_sizes[i] 
      dilation = dilations[i]
      
      print("--- kernel_size {}".format(kernel_size))
      print("--- dilation {}".format(dilation))
      
      c = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(s)
      # 2023/06/20
      if normalization:
        c = BatchNormalization()(c) 
      c = Dropout(dropout_rate * i)(c)
      c = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(c)
      # 2023/06/25
      if normalization:
        c = BatchNormalization()(c) 
      if i < (num_layers-1):
        p = MaxPool2D(pool_size=pool_size)(c)
        s = p
      enc.append(c)
    
    enc_len = len(enc)
    enc.reverse()
    n = 0
    c = enc[n]
    
    # --- Decoder
    for i in range(num_layers-1):
      kernel_size = rkernel_sizes[i] 
      dilation = rdilations[i]
      print("+++ kernel_size {}".format(kernel_size))
      print("+++ dilation {}".format(dilation))

      f = enc_len - 2 - i
      filters = base_filters* (2**f)
      u = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c)
      n += 1
      u = concatenate([u, enc[n]])
      u = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      # 2023/06/20
      if normalization:
        u = BatchNormalization()(u) 
      u = Dropout(dropout_rate * f)(u)
      u = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      # 2023/06/25
      if normalization:
        u = BatchNormalization()(u) 
      c  = u

    # outouts
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c)

    # create Model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model
  
  def create_dirs(self, eval_dir, model_dir ):
    # 2023/06/20
    dt_now = str(datetime.datetime.now())
    dt_now = dt_now.replace(":", "_").replace(" ", "_")
    create_backup = self.config.get(TRAIN, "create_backup", False)
    if os.path.exists(eval_dir):
      # if create_backup flag is True, move previous eval_dir to *_bak  
      if create_backup:
        moved_dir = eval_dir +"_" + dt_now + "_bak"
        shutil.move(eval_dir, moved_dir)
        print("--- Moved to {}".format(moved_dir))
      else:
        shutil.rmtree(eval_dir)

    if not os.path.exists(eval_dir):
      os.makedirs(eval_dir)

    if os.path.exists(model_dir):
      # if create_backup flag is True, move previous model_dir to *_bak  
      if create_backup:
        moved_dir = model_dir +"_" + dt_now + "_bak"
        shutil.move(model_dir, moved_dir)
        print("--- Mmoved to {}".format(moved_dir))      
      else:
        shutil.rmtree(model_dir)

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

  def train(self, x_train, y_train): 
    batch_size = self.config.get(TRAIN, "batch_size")
    epochs     = self.config.get(TRAIN, "epochs")
    patience   = self.config.get(TRAIN, "patience")
    eval_dir   = self.config.get(TRAIN, "eval_dir")
    model_dir  = self.config.get(TRAIN, "model_dir")
    metrics    = ["accuracy", "val_accuracy"]
    try:
      metrics    = self.config.get(TRAIN, "metrics")
    except:
      pass
    # 2023/06/20
    self.create_dirs(eval_dir, model_dir)
    # Copy current config_file to model_dir
    shutil.copy2(self.config_file, model_dir)
    print("-- Copied {} to {}".format(self.config_file, model_dir))
    
    weight_filepath   = os.path.join(model_dir, BEST_MODEL_FILE)

    early_stopping = EarlyStopping(patience=patience, verbose=1)
    check_point    = ModelCheckpoint(weight_filepath, verbose=1, save_best_only=True)
    epoch_change   = EpochChangeCallback(eval_dir, metrics)

    results = self.model.fit(x_train, y_train, 
                    validation_split=0.2, batch_size=batch_size, epochs=epochs, 
                    shuffle=False,
                    callbacks=[early_stopping, check_point, epoch_change],
                    verbose=1)
  # 2023/05/09
  def load_model(self) :
    rc = False
    if  not self.model_loaded:    
      model_dir  = self.config.get(TRAIN, "model_dir")
      weight_filepath = os.path.join(model_dir, BEST_MODEL_FILE)
      if os.path.exists(weight_filepath):
        self.model.load_weights(weight_filepath)
        self.model_loaded = True
        print("=== Loaded a weight_file {}".format(weight_filepath))
        rc = True
      else:
        message = "Not found a weight_file " + weight_filepath
        raise Exception(message)
    else:
      pass
      #print("== Already loaded a weight file.")
    return rc

  # 2023/05/05 Added newly.    
  def infer(self, input_dir, output_dir, expand=True):
    writer       = GrayScaleImageWriter()
    # We are intereseted in png and jpg files.
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    image_files += glob.glob(input_dir + "/*.tif")
    #2023/05/15 Added *.bmp files
    image_files += glob.glob(input_dir + "/*.bmp")

    width        = self.config.get(MODEL, "image_width")
    height       = self.config.get(MODEL, "image_height")

    merged_dir   = None
    try:
      merged_dir = self.config.get(INFER, "merged_dir")
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    except:
      pass

    for image_file in image_files:
      basename = os.path.basename(image_file)
      name     = basename.split(".")[0]
      # <fixed> 2023/07/15 to avoid error on png file
      img      = cv2.imread(image_file)
      img      = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
      # </fixed>

      h = img.shape[0]
      w = img.shape[1]
      # Any way, we have to resize input image to match the input size of our TensorflowUNet model.
      img         = cv2.resize(img, (width, height))
      predictions = self.predict([img], expand=expand)
      prediction  = predictions[0]
      image       = prediction[0]    
      # Resize the predicted image to be the original image size (w, h), and save it as a grayscale image.
      # Probably, this is a natural way for all humans. 
      mask = writer.save_resized(image, (w, h), output_dir, name)

      print("--- image_file {}".format(image_file))
      if merged_dir !=None:
        # Resize img to the original size (w, h)
        img   = cv2.resize(img, (w, h))
        img += mask
        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)

  def predict(self, images, expand=True):
    self.load_model()
    predictions = []
    for image in images:
      #print("=== Input image shape {}".format(image.shape))
      if expand:
        image = np.expand_dims(image, 0)
      pred = self.model.predict(image)
      predictions.append(pred)
    return predictions    

  # 2023/06/05
  # 1 Split the orginal image to some tiled-images
  # 2 Infer segmentation regions on those images 
  # 3 Merge detected regions into one image

  # 2023/07/01
  # Added MARGIN to cropping 

  def infer_tiles(self, input_dir, output_dir, expand=True):
    
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    image_files += glob.glob(input_dir + "/*.tif")
    image_files += glob.glob(input_dir + "/*.bmp")
    MARGIN       = self.config.get(TILEDINFER, "overlapping", dvalue=0)
    print("MARGIN {}".format(MARGIN))
    
    merged_dir   = None
    try:
      merged_dir = self.config.get(TILEDINFER, "merged_dir")
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    except:
      pass
    split_size  = self.config.get(MODEL, "image_width")
    print("---split_size {}".format(split_size))
    
    for image_file in image_files:
      image = Image.open(image_file)
      w, h  = image.size

      vert_split_num  = h // split_size
      if h % split_size != 0:
        vert_split_num += 1

      horiz_split_num = w // split_size
      if w % split_size != 0:
        horiz_split_num += 1

      bgcolor = self.config.get(TILEDINFER, "background", dvalue=0)  
      print("=== bgcolor {}".format(bgcolor))
      background = Image.new("L", (w, h), bgcolor)
      #print("=== width {} height {}".format(w, h))
      #print("=== horiz_split_num {}".format(horiz_split_num))
      #print("=== vert_split_num  {}".format(vert_split_num))
      #input("----")
  
      #input("----")
      for j in range(vert_split_num):
        for i in range(horiz_split_num):
          left  = split_size * i
          upper = split_size * j
          right = left  + split_size
          lower = upper + split_size

          if left >=w or upper >=h:
            continue 
      
          #cropped = image.crop((left, upper, right, lower))
          #2023/06/21
          left_margin  = MARGIN
          upper_margin = MARGIN
          if left-MARGIN <0:
            left_margin = 0
          if upper-MARGIN <0:
            upper_margin = 0
          
          #cropped = image.crop((left-MARGIN, upper-MARGIN, right+MARGIN, lower+MARGIN ))
          cropped = image.crop((left-left_margin, upper-upper_margin, right+MARGIN, lower+MARGIN ))

          cw, ch  = cropped.size
          cropped = cropped.resize((split_size, split_size))
          predictions = self.predict([cropped], expand=expand)
          prediction  = predictions[0]
          mask        = prediction[0]    

          img         = self.mask_to_image(mask)
          img         = img.resize((cw, ch))

          img         = img.convert("L")
          #2023/06/21
          ww, hh      = img.size
          #img         = img.crop((MARGIN, MARGIN, ww-MARGIN, hh-MARGIN))
          img         = img.crop((left_margin, upper_margin, ww-left_margin, hh-upper_margin ))
          
          ww, hh      = img.size
          background.paste(img, (left, upper))
          print("---paste j:{} i:{} ww:{} hh:{}".format(j, i, ww, hh))
          
      basename = os.path.basename(image_file)
      output_file = os.path.join(output_dir, basename)
      #input("----")
      #background = background.filter(filter=ImageFilter.BLUR)
      background.save(output_file)
      
      if merged_dir !=None:
        # Resize img to the original size (w, h)
        img   = np.array(image)
        img   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask  = np.array(background)
        mask   = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img += mask

        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)     


  def mask_to_image(self, data, factor=255.0):
    h = data.shape[0]
    w = data.shape[1]

    data = data*factor
    data = data.reshape([w, h])
    image = Image.fromarray(data)
   
    return image


  def evaluate(self, x_test, y_test): 
    self.load_model()
    score = self.model.evaluate(x_test, y_test, verbose=1)
    print("Test loss    :{}".format(round(score[0], 4)))     
    print("Test accuracy:{}".format(round(score[1], 4)))
     

  # 2023/07/10
  def inspect(self, image_file='./model.png', summary_file="./summary.txt"):
    # Please download and install graphviz for your OS
    # https://www.graphviz.org/download/ 
    tf.keras.utils.plot_model(self.model, to_file=image_file, show_shapes=True)
    print("=== Saved model graph as an image_file {}".format(image_file))
    # https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    with open(summary_file, 'w') as f:
      # Pass the file handle in as a lambda function to make it callable
      self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("=== Saved model summary as a text_file {}".format(summary_file))


if __name__ == "__main__":

  try:
    # Default config_file
    config_file    = "./train_eval_infer.config"
    # You can specify config_file on your command line parammeter.
    if len(sys.argv) == 2:
      confi_file= sys.argv[1]
      if not os.path.exists(config_file):
         raise Exception("Not found " + config_file)
     
    config   = ConfigParser(config_file)
    
    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowUNet(config_file)
    # Please download and install graphviz for your OS
    # https://www.graphviz.org/download/ 
    image_file = './asset/model.png'
    tf.keras.utils.plot_model(model.model, to_file=image_file, show_shapes=True)

  except:
    traceback.print_exc()
    
