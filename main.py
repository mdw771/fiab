from util import *
from scipy.misc import imsave
import time


img_ls = ['IMG_1158.jpg', 'IMG_1159.jpg', 'IMG_1160.jpg', 'IMG_1161.jpg']


t0= time.time()
pano = panorama(img_ls, n_pi=1000)
pano_img = pano.build_panorama(0.65, interpolate=False)
# pano.visualize_match(1)
imsave('panorama.png', pano_img)
print 'Done in {:.2f} s.'.format(time.time()-t0)