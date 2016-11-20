import numpy as np
import cv2
import gc


def pyramid_blend(img1, img2, depth=5):

    """
    Blank areas must be filled with NaN.
    """

    # prepare
    pano_img1 = np.zeros([np.max([img1.shape[0], img2.shape[0]]), np.max([img1.shape[1], img2.shape[1]]), 3])
    pano_img2 = np.zeros([np.max([img1.shape[0], img2.shape[0]]), np.max([img1.shape[1], img2.shape[1]]), 3])
    pano_img1[...] = np.nan
    pano_img1[:img1.shape[0], :img1.shape[1], :] = img1
    pano_img2[...] = np.nan
    pano_img2[:img2.shape[0], :img2.shape[1], :] = img2

    # find overlapping area
    nnls_1 = np.nonzero(np.isfinite(img1))
    nnls_2 = np.nonzero(np.isfinite(img2))
    global_corners = np.zeros([2, 2])
    global_corners[0, 0] = 0
    global_corners[0, 1] = np.min(nnls_2[1])
    global_corners[1, 0] = np.max(np.hstack([nnls_1[0], nnls_2[0]]))
    global_corners[1, 1] = np.max(nnls_1[1])
    global_corners = global_corners.astype('int')
    if (global_corners[1, 0]-global_corners[0, 0]+1) % 2 == 1:
        global_corners[1, 0] -= 1
    if (global_corners[1, 1]-global_corners[0, 1]+1) % 2 == 1:
        global_corners[1, 1] -= 1
    ovlap1 = pano_img1[global_corners[0, 0]:global_corners[1, 0]+1, global_corners[0, 1]:global_corners[1, 1]+1, :]
    nanmask1 = np.isnan(ovlap1)
    ovlap1[nanmask1] = 0
    ovlap2 = pano_img2[global_corners[0, 0]:global_corners[1, 0]+1, global_corners[0, 1]:global_corners[1, 1]+1, :]
    nanmask2 = np.isnan(ovlap2)
    ovlap2[nanmask2] = 0

    # create gaussian pyramids
    g1 = _create_gaussian_pyramid(ovlap1, depth)
    g2 = _create_gaussian_pyramid(ovlap2, depth)

    # create laplacian
    l1 = _create_laplacian_pyramid(g1, depth)
    l2 = _create_laplacian_pyramid(g2, depth)

    # blend
    lf = []
    for la, lb in zip(l1, l2):
        ls = np.hstack([la[:, :int(la.shape[1]/2), :], lb[:, int(lb.shape[1]/2):, :]])
        lf.append(ls)

    # reconstruct
    res = lf[0]
    for i in range(1, depth):
        res = cv2.pyrUp(res)
        l_upper = lf[i]
        if res.shape[0] > l_upper.shape[0]:
            res = np.delete(res, (-1), axis=0)
        if res.shape[1] > l_upper.shape[1]:
            res = np.delete(res, (-1), axis=1)
        res = res + l_upper

    # reinsert
    nanmask = nanmask1 * nanmask2
    res[nanmask] = np.nan
    pano = pano_img2.copy()
    pano[:img1.shape[0], :img1.shape[1]] = img1
    pano[global_corners[0, 0]:global_corners[1, 0]+1, global_corners[0, 1]:global_corners[1, 1]+1, :] = res
    gc.collect()
    return pano


def _create_laplacian_pyramid(g, depth):

    l = [g[depth-1]]
    for i in range(depth-1, 0, -1):
        g_expand = cv2.pyrUp(g[i])
        g_upper = g[i-1]
        if g_expand.shape[0] > g_upper.shape[0]:
            g_expand = np.delete(g_expand, (-1), axis=0)
        if g_expand.shape[1] > g_upper.shape[1]:
            g_expand = np.delete(g_expand, (-1), axis=1)
        l.append(g_upper-g_expand)
    return l


def _create_gaussian_pyramid(img, depth):

    temp = img.copy()
    g = [temp]
    for i in range(depth):
        temp = cv2.pyrDown(temp)
        g.append(temp)
    return g






