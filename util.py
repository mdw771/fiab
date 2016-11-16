import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imrotate
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import gc
import time
from itertools import izip
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()


class sift_descriptor(object):

    def __init__(self, nb_theta, nb_g, y, x):

        self.nb_theta = nb_theta
        self.nb_g = nb_g
        self.coords = (y, x)

    def get_descriptor(self, threshold):

        self.descriptor = np.zeros(128)
        bins = np.linspace(0, 2*np.pi, 9)
        for y in range(4):
            for x in range(4):
                arr_theta = self.nb_theta[y*4:(y+1)*4, x*4:(x+1)*4].flatten()
                arr_g = self.nb_g[y*4:(y+1)*4, x*4:(x+1)*4].flatten()
                arr_theta[arr_g<threshold] = 10
                local_hist = np.histogram(arr_theta, bins)
                ind = (y*4+x) * 8
                self.descriptor[ind:ind+8] = local_hist


class image(object):

    def __init__(self, img, index):

        self.img = img
        self.index = index
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.descriptors = []

    def get_sift_descriptors(self, radius, n_ip):

        g, theta = get_gradient_map(self.img_gray)
        f = get_corner_strength(self.img_gray, 3)
        feat = find_features(f, n_ip)
        self.descriptors = sift_descriptors(feat, g, theta, self.img_gray, self.descriptors)





def find_local_maxima(a):

    res = np.ones(a.shape, dtype='bool')
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i != 0 or j != 0:
                temp = np.roll(np.roll(a, i, axis=0), j, axis=1)
                res = res * (a > temp)

    return res


def get_gradient_map(img):

    # compute gradient
    img = img.astype('float32')
    di_y, di_x = np.gradient(img)

    # smoothen map
    g = cv2.getGaussianKernel(9, 1)
    g = np.dot(g, g.transpose())
    di_ys = convolve2d(di_y, g, mode='same', boundary='symm')
    di_xs = convolve2d(di_x, g, mode='same', boundary='symm')

    # compute magnitude
    mag = np.sqrt(di_ys ** 2 + di_xs ** 2)

    # compute orientation
    u_sin = np.zeros(img.shape)
    u_cos = np.zeros(img.shape)
    nan_loc = np.where(mag != 0)
    u_cos[nan_loc] = di_ys[nan_loc] / mag[nan_loc]
    u_sin[nan_loc] = di_xs[nan_loc] / mag[nan_loc]
    theta = np.arccos(u_cos)
    theta[u_sin<0] = 2*np.pi - theta[u_sin<0]

    return mag, theta


def get_corner_strength(img, radius):

    img = img.astype('float32')

    # get gradient
    print('Getting gradient')
    di_y, di_x = np.gradient(img)
    di_ys = gaussian_filter(di_y, 1)
    di_xs = gaussian_filter(di_x, 1)
    di_xx = di_xs ** 2
    di_yy = di_ys ** 2
    di_xy = di_xs * di_ys

    # get gaussian window
    dim = 2 * radius + 1
    g = cv2.getGaussianKernel(dim, 1)
    g = np.dot(g, g.transpose())

    # compute Harris matrix elements and calculate corner strength
    print('Computing Harris')
    f = np.zeros(img.shape)
    h = np.zeros([img.shape[0], img.shape[1], 2, 2])
    print('    Convolution')
    s_xx = convolve2d(di_xx, g, mode='same')
    s_yy = convolve2d(di_yy, g, mode='same')
    s_xy = convolve2d(di_xy, g, mode='same')
    print('    Build full Harris')
    h[:, :, 0, 0] = s_xx
    h[:, :, 0, 1] = s_xy
    h[:, :, 1, 0] = s_xy
    h[:, :, 1, 1] = s_yy
    print('    Compute f')
    trc = np.trace(h, axis1=2, axis2=3)
    nan_loc = np.where(trc != 0)
    det = np.linalg.det(h)
    f[nan_loc] = det[nan_loc] / trc[nan_loc]
    # f = det - 0.04*trc**2
    f[:dim, :] = 0
    f[-dim:, :] = 0
    f[:, :dim] = 0
    f[:, -dim:] = 0



    gc.collect()

    return f


def find_features(f, n_ip):

    print('Non-maximal suppression')

    # find interest points
    ip = find_local_maxima(f)
    ip = ip * (f > 10)
    f = f * ip

    # adaptive non-maximal suppression

    # mark global maximum
    feat = np.zeros(f.shape, dtype='bool')
    # ind = np.unravel_index(np.argmax(f), f.shape)
    # feat[ind] = 1

    # find minimum radii
    n_ipnz = np.count_nonzero(ip)
    if n_ipnz < n_ip:
        print('Warning: specified feature number larger than actual number of interest points.')
        n_ip = n_ipnz
    ipnz = np.nonzero(ip)
    ind_ls = np.zeros([n_ipnz, 2])
    ind_ls[:, 0] = ipnz[0]
    ind_ls[:, 1] = ipnz[1]
    ind_ls = ind_ls.astype('int')
    nb_table = np.zeros([n_ipnz, 3])
    c = 0.9
    counter = -1
    mxy = f.shape[0] - 1
    mxx = f.shape[1] - 1
    for y, x in ind_ls:
        counter += 1
        value = f[y, x] / c
        found = 0
        i = 1
        while not found:
            smlr_dist = f.shape[0]**2+f.shape[1]**2 + 1
            if y-i < 0 and y+i > mxy and x-i < 0 and x+i > mxx:
                found = 1
                smlr_dist = np.inf
                nb_table[counter, :] = [y, x, smlr_dist]
            else:
                arr = f[int(np.clip(y-i, 0, mxy)), int(np.clip(x-i, 0, mxx)):int(np.clip(x+i, 0, mxx))]
                smlr_dist = _update_dist(arr, smlr_dist, value, i)
                arr = f[int(np.clip(y-i, 0, mxy)):int(np.clip(y+i, 0, mxy)), int(np.clip(x+i, 0, mxx))]
                smlr_dist = _update_dist(arr, smlr_dist, value, i)
                arr = f[int(np.clip(y+i, 0, mxy)), int(np.clip(x-i+1, 0, mxx)):int(np.clip(x+i+1, 0, mxx))]
                smlr_dist = _update_dist(arr, smlr_dist, value, i)
                arr = f[int(np.clip(y-i+1, 0, mxy)):int(np.clip(y+i+1, 0, mxy)), int(np.clip(x-i, 0, mxx))]
                smlr_dist = _update_dist(arr, smlr_dist, value, i)
                if smlr_dist < f.shape[0]**2+f.shape[1]**2 + 1:
                    found = 1
                    nb_table[counter, :] = [y, x, smlr_dist]
                i += 1

    # sort in descending order
    nb_table = nb_table[nb_table[:, 2].argsort()[::-1]]

    # finalize feature matrix
    for i in range(n_ip):
        y, x, _ = nb_table[i, :]
        feat[y, x] = i

    return feat


def sift_descriptors(feat, g, theta, img, res_ls):

    # discard features too close to the edges
    feat[:8, :] = 0
    feat[:, -8:] = 0
    feat[-8:, :] = 0
    feat[:, :8] = 0

    # generate sift descriptors
    n_ipnz = np.count_nonzero(feat)
    ipnz = np.nonzero(feat)
    ind_ls = np.zeros([n_ipnz, 2])
    ind_ls[:, 0] = ipnz[0]
    ind_ls[:, 1] = ipnz[1]
    ind_ls = ind_ls.astype('int')
    for y, x in ind_ls:





def _update_dist(arr, smlr_dist, value, i):

    lgr_ls = arr[arr>value]
    nz_ls = np.nonzero(lgr_ls)[0]
    for ii in nz_ls:
        dist = (ii-i)**2 + i**2
        if dist < smlr_dist:
            smlr_dist = dist
    return smlr_dist



