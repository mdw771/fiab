from util import *
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave
from scipy.signal import convolve2d
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from mpi4py import MPI


img_ls = ['IMG_1158.jpg', 'IMG_1159.jpg', 'IMG_1160.jpg', 'IMG_1161.jpg']


pano = panorama(img_ls, 1000)
pano_img = pano.build_panorama(0.65, interpolate=False)
# pano.visualize_match(1)
imsave('panorama.png', pano_img)