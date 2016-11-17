from util import *
import numpy as np
from scipy.ndimage import imread
from scipy.signal import convolve2d
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from mpi4py import MPI


img_ls = ['01.tif', '02.tif']


pano = panorama(img_ls, 500)
pano.match_panorama(0.6)
pano.visualize_match(0)
