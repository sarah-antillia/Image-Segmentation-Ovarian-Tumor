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

# TensorflowAttentionUNetBrainTumorTrainer.py
# 2023/05/30 to-arai

# This is based on the code in the following web sites:

# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset

from TensorflowUNet import TensorflowUNet

MODEL  = "model"
TRAIN  = "train"


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]

    config   = ConfigParser(config_file)

    # Create a UNetMolde and compile
    model   = TensorflowUNet(config_file)
    
    dataset = ImageMaskDataset(config_file)
    x_train, y_train = dataset.create(dataset=TRAIN)

    model.train(x_train, y_train)

  except:
    traceback.print_exc()
    
