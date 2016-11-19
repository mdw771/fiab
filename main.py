from util import *
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave
from scipy.signal import convolve2d
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from mpi4py import MPI


__author__ = 'Ming Du'
__email__ = 'mingdu2015@u.northwestern.edu'


img_ls = ['01.tif', '02.tif', '03.tif']
# img_ls = ['0100.tiff', '0101.tiff']


pano = panorama(img_ls, 1000)
pano_img = pano.build_panorama(0.65, interpolate=False)
# pano.visualize_match(1)
imsave('panorama.png', pano_img)