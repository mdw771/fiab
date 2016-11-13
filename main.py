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

img = imread('01.JPG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t0 = time.time()
feat, _ = find_feature(img, 3)
print('Time: ', time.time()-t0)

img[feat] = 0

#gy, gx = np.gradient(img)
plt.figure(1)
plt.imshow(img, cmap='gray')
plt.show()