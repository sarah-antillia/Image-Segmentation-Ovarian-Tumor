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

# 2023/07/28
# ImageMaskDatasetGenerator.py

import os
import glob
from re import A
import shutil
from PIL import Image, ImageOps, ImageFilter
import cv2
import traceback
import numpy as np

class ImageMaskDatasetGenerator:
  
  def __init__(self, resize=256):
    self.RESIZE   = resize
    self.blur_mask = True


  def augment(self, image, output_dir, filename):
    # 2023/07/27
    #ANGLES = [30, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    ANGLES = [0, 90, 180, 270]
    for angle in ANGLES:
      rotated_image = image.rotate(angle)
      output_filename = "rotated_" + str(angle) + "_" + filename
      rotated_image_file = os.path.join(output_dir, output_filename)
      #cropped  =  self.crop_image(rotated_image)
      rotated_image.save(rotated_image_file)
      print("=== Saved {}".format(rotated_image_file))
    # Create mirrored image
    mirrored = ImageOps.mirror(image)
    output_filename = "mirrored_" + filename
    image_filepath = os.path.join(output_dir, output_filename)
    #cropped = self.crop_image(mirrored)
    mirrored.save(image_filepath)
    print("=== Saved {}".format(image_filepath))
        
    # Create flipped image
    flipped = ImageOps.flip(image)
    output_filename = "flipped_" + filename

    image_filepath = os.path.join(output_dir, output_filename)
    #cropped = self.crop_image(flipped)

    flipped.save(image_filepath)
    print("=== Saved {}".format(image_filepath))


  def resize_to_square(self, image):
     w, h  = image.size

     bigger = w
     if h > bigger:
       bigger = h

     background = Image.new("RGB", (bigger, bigger), (0, 0, 0))
    
     x = (bigger - w) // 2
     y = (bigger - h) // 2
     background.paste(image, (x, y))
     background = background.resize((self.RESIZE, self.RESIZE))

     return background
  

  def create(self, input_images_dir, input_masks_dir,  output_dir,
                            debug=False):
    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    image_files = glob.glob(input_images_dir + "/*.JPG")
    
    if image_files == None or len(image_files) == 0:
      print("FATAL ERROR: Not found mask files")
      return

    for image_file in image_files:
      basename = os.path.basename(image_file)
      image = Image.open(image_file).convert("RGB")
      pngname = basename.replace(".JPG", ".PNG")
      mask_filepath = os.path.join(input_masks_dir, pngname)
      print("--- maskfilepath {}".format(mask_filepath))
      mask = Image.open(mask_filepath).convert("RGB")
      #w, h = image.size
      #image = image.resize((w, h))
      
      basename = basename.replace(".JPG", ".jpg")
      image_output_filepath = os.path.join(output_images_dir, basename)
      
      squared_image = self.resize_to_square(image)
      # Save the cropped_square_image
      #cropped = self.crop_image(squared_image)
      squared_image.save(image_output_filepath)
      print("--- Saved cropped_square_image {}".format(image_output_filepath))

      self.augment(squared_image, output_images_dir, basename)
   
      #print("--- mask_file {}".format(mask_file)) 

      mask_color = (255, 255, 255)
      xmask = self.create_mono_color_mask(mask, mask_color= mask_color)
   
      # Blur mask 
      if self.blur_mask:
        print("---blurred ")
        xmask = xmask.filter(ImageFilter.BLUR)
      
      if debug:
        xmask.show()
        input("XX")   
      out_mask_file = image_file
      mask_output_filepath = os.path.join(output_masks_dir, basename)

      squared_mask = self.resize_to_square(xmask)
      #cropped_mask = self.crop_image(squared_mask)
      squared_mask.save(mask_output_filepath)

      print("--- Saved cropped_squared_mask {}".format(mask_output_filepath))
      self.augment(squared_mask, output_masks_dir, basename)


  def create_mono_color_mask(self, mask, mask_color=(255, 255, 255)):
    rw, rh = mask.size    
    xmask = Image.new("RGB", (rw, rh))
    #print("---w {} h {}".format(rw, rh))

    for i in range(rw):
      for j in range(rh):
        color = mask.getpixel((i, j))
        (r, g, b) = color
        # If color is blue
        if r>20 or g >20 or b > 20:
          xmask.putpixel((i, j), mask_color)

    return xmask
  

if __name__ == "__main__":
  try:
   
    input_images_dir = "./OTU_2d/images/"
    input_masks_dir  = "./OTU_2d/annotations/"
    

    generator = ImageMaskDatasetGenerator()
    
    output_dir = "./Ovarian-Tumor-master"
    generator.create(input_images_dir, input_masks_dir, output_dir, debug=False)

  except:
    traceback.print_exc()
    pass
