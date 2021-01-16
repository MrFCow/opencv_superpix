import io
import sys
import base64
import traceback
import numpy as np

import skimage
import skimage.io
import skimage.segmentation
from skimage.measure import regionprops

class ImageProcessor:
  def __init__(self):
    pass

  def convert_image_b64string_to_ndarray(self, base64_image):
    encoded = base64.b64decode(base64_image)
    image_array = skimage.io.imread(io.BytesIO(encoded))[:,:,:3] # y,x,channel, if png, drop alpha
    return image_array

  def produce_segments(self, image_array, n_segments=10, sigma=3):
    try:
      segments = skimage.segmentation.slic(image_array, n_segments = n_segments, sigma = sigma)
      regions = regionprops(segments+1)
      centroids = [props.centroid for props in regions]
      return {
          "segments": segments,
          "centroids": centroids
      }
    except Exception as err:
      traceback.print_exception(*sys.exc_info())

  def __call__(self, base64_image, n_segments=10, sigma=3):
    img_array = self.convert_image_b64string_to_ndarray(base64_image)
    result = self.produce_segments(img_array, n_segments=10, sigma=3)
    return result