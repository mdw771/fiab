from util import *
import numpy as np
from scipy.ndimage import imread
from scipy.signal import convolve2d
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()


def align_image(img):

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return

img = imread('01.tif')
# img = imread('window.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t0 = time.time()
g, theta = get_gradient_map(img)
feat = get_corner_strength(img, 3)
feat = find_features(feat, 250)
# feat = cv2.cornerHarris(img,2,3,0.04)

print('Time: ', time.time()-t0)

#feat = cv2.dilate(feat.astype('float32'), np.ones([3,3])).astype('bool')
img[feat] = 255


plt.figure(1)
plt.imshow(img, cmap='gray', interpolation='nearest')

plt.figure(2)
plt.imshow(feat, cmap='gray', interpolation='nearest')

plt.show()