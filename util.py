from blend import *
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import imread
from scipy.interpolate import interp2d
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import gc
import random
import time
import copy
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()


class panorama(object):

    def __init__(self, img_ls, n_pi):

        self.img_ls = []
        for index in range(len(img_ls)):
            img = imread(img_ls[index])
            img = image(img, index)
            img.get_sift_descriptors(3, n_pi)
            self.img_ls.append(img)
        self.n_img = len(img_ls)

    def build_panorama(self, match_threshold, interpolate=True, pyramid_depth=5):

        pano = copy.copy(self.img_ls[0])
        for img in self.img_ls[1:]:
            print('Building panorama: {:d}'.format(img.index))
            img.match_img(pano, match_threshold)
            h = img.get_projection_matrix(pano, n_iter=1000, margin=1)

            # determine range of warpped image
            corners = np.array([[0, 0, img.shape[0]-1, img.shape[0]-1],
                                [0, img.shape[1]-1, 0, img.shape[1]-1],
                                [1, 1, 1, 1]])
            corners = np.dot(h, corners)
            corners[0, :] = corners[0, :] / corners[2, :]
            corners[1, :] = corners[1, :] / corners[2, :]
            true_corners = np.array([[np.min(corners[0, :]), np.min(corners[1, :])],
                                     [np.max(corners[0, :]), np.max(corners[1, :])]])
            corners = np.array([[np.min(np.append(corners[0, :], 0)), np.min(np.append(corners[1, :], 0))],
                                [np.max(corners[0, :]), np.max(corners[1, :])]])
            corners[:, 0] = np.clip(corners[:, 0], 0, np.inf)
            true_corners[:, 0] = np.clip(true_corners[:, 0], 0, np.inf)
            corners[:, 1] = np.clip(corners[:, 1], 0, np.inf)
            true_corners[:, 1] = np.clip(true_corners[:, 1], 0, np.inf)
            corners = corners.astype('int')
            true_corners = true_corners.astype('int')

            # map back to unwrapped image space
            newshape = (corners[1, 0]-corners[0, 0]+1, corners[1, 1]-corners[0, 1]+1)
            yrange = range(true_corners[0, 0], true_corners[1, 0]+1)
            xrange = range(true_corners[0, 1], true_corners[1, 1]+1)
            xx, yy = np.meshgrid(xrange, yrange)
            p = np.zeros([3, xx.size])
            p[0, :] = yy.reshape([1, yy.size])
            p[1, :] = xx.reshape([1, xx.size])
            p[2, :] = np.ones(xx.size)
            p0 = np.dot(np.linalg.inv(h), p)
            p0[0, :] = p0[0, :] / p0[2, :]
            p0[1, :] = p0[1, :] / p0[2, :]
            print('    Rebuilding warped image')
            if interpolate:
                f0 = interp2d(range(img.shape[1]), range(img.shape[0]), img.img[:, :, 0], fill_value=0)
                f1 = interp2d(range(img.shape[1]), range(img.shape[0]), img.img[:, :, 1], fill_value=0)
                f2 = interp2d(range(img.shape[1]), range(img.shape[0]), img.img[:, :, 2], fill_value=0)
            pano_img = np.zeros([np.max([pano.shape[0], newshape[0]]), np.max([pano.shape[1], newshape[1]]), 3])
            pano_img[...] = np.nan
            coords = np.round(np.array([p0[0, :], p0[1, :], p[0, :], p[1, :]]).transpose())
            for y0, x0, y, x in coords:
                if 0 < y0 < img.shape[0] and 0 < x0 < img.shape[1]:
                    if interpolate:
                        value0 = f0(x0, y0)[0]
                        value1 = f1(x0, y0)[0]
                        value2 = f2(x0, y0)[0]
                        value = np.array([value0, value1, value2])
                    else:
                        value = img.img[int(y0), int(x0), :]
                    pano_img[int(y), int(x), :] = value

            # blend
            print('Blending')
            pano_img = pyramid_blend(pano.img, pano_img, depth=pyramid_depth)
            des_temp = pano.descriptors
            mdes_temp = pano.matched_descriptors
            pano = image(pano_img, 0)
            pano.descriptors = des_temp
            pano.matched_descriptors = mdes_temp

            # update feature coordinates
            print('Updating feature coordinates in pano object')
            for f in img.descriptors:
                ff = copy.copy(f)
                p0 = np.array([[ff.coords[0]], [ff.coords[1]], [1]])
                p = np.dot(h, p0)
                p[0, 0] = p[0, 0] / p[2, 0]
                p[1, 0] = p[1, 0] / p[2, 0]
                ff.coords = (p[0, 0], p[1, 0])
                pano.descriptors.append(ff)

        pano.img[np.isnan(pano.img)] = 0
        return pano.img

    def visualize_match(self, index):

        img1 = self.img_ls[index]
        img2 = self.img_ls[index-1]
        full = np.zeros([np.max([img1.shape[0], img2.shape[0]]), img1.shape[1]+img2.shape[1]+10])
        full[:img1.shape[0], :img1.shape[1]] = img1.img_gray
        offset_x = img1.shape[1] + 10
        full[:img2.shape[0], offset_x:offset_x+img2.shape[1]] = img2.img_gray
        plt.figure()
        plt.imshow(full, cmap='gray')
        for f1 in img1.descriptors:
            if f1.nn1 is not None:
                y1, x1 = f1.coords
                y2, x2 = f1.nn1.coords
                x2 += offset_x
                plt.plot([x1, x2], [y1, y2])
        plt.show()


class image(object):

    def __init__(self, img, index):

        self.index = index
        if img.ndim == 2:
            self.img = np.zeros([img.shape[0], img.shape[1], 3])
            self.img[:, :, 0] = img
            self.img[:, :, 1] = img
            self.img[:, :, 2] = img
            self.img_gray = img
            self.img_gray = (self.img_gray - self.img_gray.min()) / (self.img_gray.max() - self.img_gray.min()) * 255
            self.img_gray = self.img_gray.astype('uint8')
        elif img.ndim == 3:
            self.img = img
            self.img_gray = cv2.cvtColor(img.astype('uint16'), cv2.COLOR_BGR2GRAY)
        self.descriptors = []
        self.shape = img.shape
        self.matched_descriptors = []
        self.size = int(self.img.size/3)

    def get_sift_descriptors(self, radius, n_ip):

        g, theta = get_gradient_map(self.img_gray)
        f = get_corner_strength(self.img_gray, radius)
        feat = find_features(f, n_ip)
        self.descriptors = find_sift_descriptors(feat, g, theta, self.descriptors, self.index)

    def match_img(self, image2, threshold):

        assert isinstance(image2, image)
        for f1 in self.descriptors:
            nn1_ssd = np.inf
            nn2_ssd = np.inf
            nn1 = None
            for f2 in image2.descriptors:
                ssd = f1.ssd(f2)
                if ssd < nn1_ssd:
                    nn1_ssd = ssd
                    nn1 = f2
                elif ssd < nn2_ssd:
                    nn2_ssd = ssd
            ssd_ratio = nn1_ssd / nn2_ssd
            if ssd_ratio < threshold:
                f1.set_best(nn1, ssd_ratio)
                self.matched_descriptors.append(f1)

    def get_projection_matrix(self, img0, n_iter=1000, margin=1):

        assert isinstance(img0, image)
        inliners = []
        if len(self.matched_descriptors) == 0:
            raise ValueError('Current image object must be matched.')
        else:
            max_inliners = 0
            # n_iter = int(np.min([n_iter, comb(len(self.matched_descriptors), 4)]))
            x0_full = np.zeros([3, len(self.matched_descriptors)])
            x1_full = np.zeros([3, len(self.matched_descriptors)])
            for i in range(len(self.matched_descriptors)):
                x0_full[0, i] = self.matched_descriptors[i].coords[0]
                x0_full[1, i] = self.matched_descriptors[i].coords[1]
                x1_full[0, i] = self.matched_descriptors[i].nn1.coords[0]
                x1_full[1, i] = self.matched_descriptors[i].nn1.coords[1]
            x0_full[2, :] = 1
            x1_full[2, :] = 1
            for i in range(n_iter):
                try:
                    temp = []
                    ransac_pairs = random.sample(self.matched_descriptors, 4)
                    x0 = np.array([ransac_pairs[0].coords])
                    x1 = np.array([ransac_pairs[0].nn1.coords])
                    for ii in range(1, len(ransac_pairs)):
                        x0 = np.append(x0, [ransac_pairs[ii].coords], axis=0)
                        x1 = np.append(x1, [ransac_pairs[ii].nn1.coords], axis=0)
                    h = _compute_projection_matrix_exact(x0, x1)
                    temp = np.dot(h, x0_full)
                    temp[2, :][temp[2, :]==0] = 1e-10
                    temp[0, :] = temp[0, :] / temp[2, :]
                    temp[1, :] = temp[1, :] / temp[2, :]
                    diff = (x1_full-temp)**2
                    diff = np.sqrt(diff[0, :]+diff[1, :])
                    if np.count_nonzero(diff < margin) > max_inliners:
                        max_inliners = np.count_nonzero(diff < margin)
                        inliners = []
                        nz = np.nonzero(diff < margin)[0]
                        for ind in nz:
                            inliners.append(self.matched_descriptors[ind])
                except:
                    continue
            print('    Number of feature pairs used for LSQ is '+str(len(inliners)))
            x0 = np.array([inliners[0].coords])
            x1 = np.array([inliners[0].nn1.coords])
            for ii in range(1, len(inliners)):
                x0 = np.append(x0, [inliners[ii].coords], axis=0)
                x1 = np.append(x1, [inliners[ii].nn1.coords], axis=0)
            h = _compute_projection_matrix_lstsq(x0, x1)
            return h


class sift_descriptor(object):

    def __init__(self, nb_g, nb_theta, y, x, img_index):

        self.nb_theta = nb_theta
        self.nb_g = nb_g
        self.coords = (y, x)
        self.vector = np.zeros(128)
        self.nn1 = None
        self.ssd_ratio = None
        self.img_index = img_index

    def get_descriptor_vector(self, threshold):

        bins = np.linspace(0, 2*np.pi, 9)
        for y in range(4):
            for x in range(4):
                arr_theta = self.nb_theta[y*4:(y+1)*4, x*4:(x+1)*4].flatten()
                arr_g = self.nb_g[y*4:(y+1)*4, x*4:(x+1)*4].flatten()
                arr_theta[arr_g<threshold] = 10
                local_hist, _ = np.histogram(arr_theta, bins)
                ind = (y*4+x) * 8
                self.vector[ind:ind+8] = local_hist

    def ssd(self, f2):

        assert isinstance(f2, sift_descriptor)
        ssd = np.sum((self.vector-f2.vector)**2)
        return ssd

    def set_best(self, f2, ssd_ratio):

        assert isinstance(f2, sift_descriptor)
        self.nn1 = f2
        self.ssd_ratio = ssd_ratio


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
        feat[int(y), int(x)] = i

    return feat


def find_sift_descriptors(feat, g, theta, res_ls, index):

    print('SIFT descriptors')

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
        feat_angle = theta[y, x]
        nb_theta = theta[y-8:y+8, x-8:x+8]
        nb_mag = g[y-8:y+8, x-8:x+8]
        # unify neighborhood orientation according to feature pixel
        nb_theta = nb_theta - feat_angle
        nb_theta[nb_theta<0] = 2*np.pi + nb_theta[nb_theta<0]
        nb_theta = np.clip(nb_theta, 0, 2*np.pi)
        sift = sift_descriptor(nb_mag, nb_theta, y, x, index)
        sift.get_descriptor_vector(0.5)
        res_ls.append(sift)

    return res_ls


def _update_dist(arr, smlr_dist, value, i):

    lgr_ls = arr[arr>value]
    nz_ls = np.nonzero(lgr_ls)[0]
    for ii in nz_ls:
        dist = (ii-i)**2 + i**2
        if dist < smlr_dist:
            smlr_dist = dist
    return smlr_dist


def _compute_projection_matrix_exact(x0, x1):

    assert isinstance(x0, np.ndarray) and isinstance(x1, np.ndarray)
    a = np.array([[x0[0, 0], x0[0, 1], 1, 0, 0, 0, -x1[0, 0]*x0[0, 0], -x1[0, 0]*x0[0, 1]],
                  [x0[1, 0], x0[1, 1], 1, 0, 0, 0, -x1[1, 0]*x0[1, 0], -x1[1, 0]*x0[1, 1]],
                  [x0[2, 0], x0[2, 1], 1, 0, 0, 0, -x1[2, 0]*x0[2, 0], -x1[2, 0]*x0[2, 1]],
                  [x0[3, 0], x0[3, 1], 1, 0, 0, 0, -x1[3, 0]*x0[3, 0], -x1[3, 0]*x0[3, 1]],
                  [0, 0, 0, x0[0, 0], x0[0, 1], 1, -x1[0, 1]*x0[0, 0], -x1[0, 1]*x0[0, 1]],
                  [0, 0, 0, x0[1, 0], x0[1, 1], 1, -x1[1, 1]*x0[1, 0], -x1[1, 1]*x0[1, 1]],
                  [0, 0, 0, x0[2, 0], x0[2, 1], 1, -x1[2, 1]*x0[2, 0], -x1[2, 1]*x0[2, 1]],
                  [0, 0, 0, x0[3, 0], x0[3, 1], 1, -x1[3, 1]*x0[3, 0], -x1[3, 1]*x0[3, 1]]])
    b = np.array([x1[0, 0], x1[1, 0], x1[2, 0], x1[3, 0], x1[0, 1], x1[1, 1], x1[2, 1], x1[3, 1]])
    h = np.linalg.solve(a, b)
    h = np.append(h, 1)
    return h.reshape([3, 3])


def _compute_projection_matrix_lstsq(x0, x1):

    assert isinstance(x0, np.ndarray) and isinstance(x1, np.ndarray)
    a = np.zeros([2*x0.shape[0], 8])
    for i in range(x0.shape[0]):
        a[i, :] = [x0[i, 0], x0[i, 1], 1, 0, 0, 0, -x1[i, 0]*x0[i, 0], -x1[i, 0]*x0[i, 1]]
    for i in range(x0.shape[0]):
        a[i+x0.shape[0], :] = [0, 0, 0, x0[i, 0], x0[i, 1], 1, -x1[i, 1]*x0[i, 0], -x1[i, 1]*x0[i, 1]]
    b = np.append(x1[:, 0], x1[:, 1])
    h = np.dot(np.linalg.pinv(a), b)
    h = np.append(h, 1)
    return h.reshape([3, 3])


def _ravel_array_index(y, x, shape):

    y[y < 0] = -1
    y[y >= shape[0]] = -1
    x[x < 0] = -1
    x[x >= shape[1]] = -1
    ind = y*shape[0] + x
    return ind
