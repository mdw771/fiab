import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
import cv2
import gc
import time
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()


def find_local_maxima(a):

    n = np.zeros([8, a.shape[0], a.shape[1]])
    ind = 0
    res = np.ones(a.shape, dtype='bool')
    for i in range(2):
        for j in range(2):
            if i != 1 and j != 1:
                m = 1 - j
                n = 1 - j
                temp = np.roll(np.roll(a, m, axis=0), n, axis=1)
                res = res * (a>temp)
                ind += 1

    return res


def find_feature(img, radius):

    dim = 2 * radius + 1

    # get gradient
    print('Getting gradient')
    di_y, di_x = np.gradient(img)
    di_xx = di_x ** 2
    di_yy = di_y ** 2
    di_xy = di_x * di_y

    # compute Harris matrix elements and calculate corner strength
    print('Computing Harris')
    p_xx = np.zeros([img.shape[0], img.shape[1], dim, dim])
    p_yy = np.zeros([img.shape[0], img.shape[1], dim, dim])
    p_xy = np.zeros([img.shape[0], img.shape[1], dim, dim])
    print('    Building P matrix')

    g = cv2.getGaussianKernel(dim, 1)
    g = np.dot(g, g.transpose())
    f = np.zeros(img.shape)
    for y in range(radius, img.shape[0]-radius):
        for x in range(radius, img.shape[1]-radius):
            p_xx = di_xx[y-radius:y+radius+1, x-radius:x+radius+1]
            p_yy = di_yy[y-radius:y+radius+1, x-radius:x+radius+1]
            p_xy = di_xy[y-radius:y+radius+1, x-radius:x+radius+1]
            h_xx = np.sum(p_xx*g)
            h_yy = np.sum(p_yy*g)
            h_xy = np.sum(p_xy*g)
            h = np.array([[h_xx, h_xy], [h_xy, h_yy]])
            t = np.trace(h)
            if t != 0:
                f[y, x] = np.linalg.det(h) / t
            else:
                f[y, x] = 0

    # compute corner strength
    print('Computing corner strength')
    f = np.linalg.det(h) / np.trace(h, axis1=2, axis2=3)
    f[:dim, :] = 0
    f[-dim:, :] = 0
    f[:, :dim] = 0
    f[:, -dim:] = 0

    # find strong corners
    feat = find_local_maxima(f)
    feat = feat * (f>10)

    # compute orientation
    g = cv2.getGaussianKernel(9, 1)
    g = np.dot(g, g.transpose())
    di_ys = convolve2d(di_y, g, mode='same', boundary='symm')
    di_xs = convolve2d(di_x, g, mode='same', boundary='symm')
    u = np.zeros([img.shape[0], img.shape[1], 2])
    u_norm = np.sqrt(di_ys**2 + di_xs**2)
    u[:, :, 0] = di_ys / u_norm
    u[:, :, 1] = di_xs / u_norm

    gc.collect()

    return feat, u
