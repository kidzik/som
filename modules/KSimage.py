import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
# import pyfftw

import copy
from scipy import misc
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter, laplace
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import distance_transform_cdt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy import cluster
from skimage import feature
from skimage import transform
from skimage.feature import peak_local_max
from skimage.filters import rank
from skimage.morphology import remove_small_objects, skeletonize
from sklearn.cluster import KMeans

from scipy.signal import fftconvolve


###########################################
def imread(image_path):
    """
    read an image
    :param image_path:
    :return:
    """
    I = cv2.imread(image_path, -1)
    # I = misc.imread(image_path)
    if I.ndim == 3:
        I = I[..., ::-1]
    if I.ndim == 2:
        I = np.expand_dims(I, 2)
    return I


###########################################
def imwrite(I, save_path):
    """
    write an image to path
    :param save_path:
    :return:
    """
    if I.ndim == 3 and I.shape[2] == 1:
        I = np.squeeze(I, axis=2)

    if I.ndim == 3 and I.shape[2] == 3:
        I = I[..., ::-1]

    # misc.imsave(save_path, I)
    cv2.imwrite(save_path, I)


###########################################
def imshow(I):
    if I.ndim == 3:
        if I.shape[2] == 1:
            I = np.squeeze(I, axis=2)

    plt.imshow(I)
    # plt.show()


###########################################
def distance_transform(binary_image):
    bw = np.logical_not(binary_image)
    D = ndimage.distance_transform_edt(bw)
    return D


###########################################
def distance_transfrom_chessboard(binary_image):
    bw = np.logical_not(binary_image)
    D = distance_transform_cdt(bw)
    return D


###########################################
def rgb2gray(rgb):
    # return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


###########################################
def auto_canny(image, sigma=0.33):
    # preprocessing
    image = ndimage.gaussian_filter(image, 1)

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = feature.canny(image, sigma=1, low_threshold=lower, high_threshold=upper)

    # return the edged image
    return edged


###########################################
def bwlabel(bw):
    if bw.dtype != 'bool':
        bw = bw.astype(np.bool)

    labels = ndimage.label(bw)
    return labels[0], labels[1]


###########################################
def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image
    """

    if n not in (4, 8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows, cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows, cols))
    south = np.zeros((rows, cols))
    west = np.zeros((rows, cols))
    east = np.zeros((rows, cols))

    north[:-1, :] = bw[1:, :]
    south[1:, :] = bw[:-1, :]
    west[:, :-1] = bw[:, 1:]
    east[:, 1:] = bw[:, :-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west == bw) & \
          (east == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:] = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:] = bw[:-1, :-1]
        south_west[1:, :-1] = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw


####################################################
def shearing(img_file):
    # Load the image as a matrix
    image = imread(img_file)

    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=0.2)

    # Apply transform to image data
    modified = transform.warp(image, afine_tf)

    return modified


#####################################################
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if len(image.shape) == 2:
        shape = image.shape[0:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        image = map_coordinates(image, indices, order=1).reshape(shape)
    else:

        shape = image.shape[0:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        for i in range(image.ndim):
            image[:, :, i] = map_coordinates(image[:, :, i], indices, order=1).reshape(shape)

    return image


#####################################################
def imresize(image, size, mode=None):
    # if image.ndim == 3 and image.shape[2] == 1:
    #     image = np.squeeze(image,2)
    resize_img = list()
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)

    for i in range(image.shape[2]):
        resize_img.append(misc.imresize(image[:, :, i], size, interp='bicubic', mode=mode))

    resize_img = np.dstack(resize_img)

    if resize_img.shape[2] == 1:
        resize_img = np.squeeze(resize_img, axis=2)

    return resize_img


#####################################################
def adaptive_histeq(img):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = list()

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    for i in range(img.shape[2]):
        cl.append(clahe.apply(img[:, :, i]))

    img = np.dstack(cl)

    if img.ndim == 3 and img.shape[img.ndim - 1] == 1:
        img = np.squeeze(img, axis=2)

    return img


#####################################################
def find_local_maxima(img, min_distance, threshold_rel):
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = np.squeeze(img)

    sigma = 1
    kernel_size = 6 * sigma - 1
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    coordinate = peak_local_max(img, min_distance=min_distance, threshold_rel=threshold_rel)

    if coordinate.size:
        if coordinate.shape[0] > 1:
            Z = cluster.hierarchy.ward(coordinate)
            cutree = cluster.hierarchy.cut_tree(Z, height=min_distance)
            unique_idx = np.unique(cutree)

            tmp_coordinate = list()
            for idx in unique_idx:
                tmp_coordinate.append(np.mean(coordinate[np.squeeze(cutree == idx), :], axis=0))

            coordinate = np.vstack(tmp_coordinate)
    #################################################################################
    # nparts = np.int(np.ceil(coordinate.shape[0] / 1000.0))
    # kmeans = KMeans(n_clusters=nparts, n_init = 1).fit(coordinate)
    # idx = kmeans.labels_
    #
    # all_coordinate = list()
    #
    # for i in range(nparts):
    #     temp_coordinate = coordinate[np.squeeze(np.argwhere(idx == i)),:]
    #
    #     for min_distance in np.linspace(int(min_distance/2.0),min_distance+1,3):
    #
    #         dist_mat = squareform(pdist(temp_coordinate, 'euclidean'))
    #
    #         dBinary = dist_mat < min_distance
    #         dBinary[np.eye(dBinary.shape[0]).astype(np.bool)] = 0
    #
    #         m = ESLclique(dBinary)
    #         m1 = m.tocsc()
    #
    #         nCliques = m1.shape[1]
    #
    #         Ic = np.zeros((nCliques, 1))
    #         Jc = np.zeros((nCliques, 1))
    #
    #         for i in range(nCliques):
    #             Im = temp_coordinate[np.squeeze(m1[:, i].toarray().astype(np.bool)),0]
    #             Jm = temp_coordinate[np.squeeze(m1[:, i].toarray().astype(np.bool)),1]
    #
    #             weight = list()
    #             for x,y in zip(Im,Jm):
    #                 weight.append(img[int(x),int(y)])
    #
    #             idx_w = np.argmax(np.array(weight))
    #
    #             Ic[i] = np.mean(Im[idx_w])
    #             Jc[i] = np.mean(Jm[idx_w])
    #
    #             # Ic[i] = np.mean(Im)
    #             # Jc[i] = np.mean(Jm)
    #
    #         Ic = np.round(Ic)
    #         Jc = np.round(Jc)
    #
    #         temp_coordinate = np.column_stack((Ic,Jc))
    #
    #     all_coordinate.append(temp_coordinate)
    #
    # coordinate = np.row_stack(all_coordinate)
    ##################################################################################

    return coordinate.astype(np.int)


#####################################################
def bwareaopen(mask, area_limit):
    mask = mask.astype(np.bool)
    # mask = mask.astype(np.uint8) * 255
    # im, contours, hierarchy = cv2.findContours(mask, 1, 2)
    #
    # temp_mask = np.zeros(mask.shape[:2], dtype='uint8')
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > area_limit:
    #         cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
    if not np.all(mask):
        temp_mask = remove_small_objects(mask, area_limit)
    else:
        temp_mask = mask

    return temp_mask.astype(np.bool)


#####################################################
def imdilate(bw, r):
    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))

    if bw.dtype == 'bool':
        bw = bw * 255
        bw = bw.astype(np.uint8)

    # return rank.maximum(bw, selem).astype(np.bool)
    return cv2.dilate(bw, selem, iterations=1).astype(np.bool)


#####################################################
def imerode(bw, r):
    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))

    if bw.dtype == 'bool':
        bw = bw * 255
        bw = bw.astype(np.uint8)

    return cv2.erode(bw, selem, iterations=1).astype(np.bool)


#####################################################
def imclose(bw, r):
    salem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))

    if bw.dtype == 'bool':
        bw = bw * 255
        bw = bw.astype(np.uint8)

    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, salem).astype(np.bool)


#####################################################
def imopen(bw, r):
    salem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))

    if bw.dtype == 'bool':
        bw = bw * 255
        bw = bw.astype(np.uint8)

    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, salem).astype(np.bool)


#####################################################
def label2idx(L):
    unq, unq_inv, unq_cnt = np.unique(L, return_inverse=True, return_counts=True)
    return dict(zip(unq, np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))))


#####################################################
def rgb2hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


#####################################################
def hsv2rgb(hsv):
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


#####################################################
def time_conv2(obssize, refsize):
    K = 2.7e-8
    time = K * np.prod(obssize) * np.prod(refsize)
    return time


def time_fft2(outsize):
    R = outsize[0]
    S = outsize[1]
    K_fft = 3.3e-7
    Tr = K_fft * R * np.log(R)

    if S == R:
        Ts = Tr
    else:
        Ts = K_fft * S * np.log(S)

    time = S * Tr + R * Ts
    return time


# def freqxcorr(a, b, outsize):
#     # Fa = np.fft.fft2(np.rot90(np.rot90(a)), [outsize[0], outsize[1]])
#     # Fb = np.fft.fft2(b, [outsize[0], outsize[1]])
#     # xcorr_ab = np.fft.ifft2(Fa * Fb)
#
#     Fa = pyfftw.interfaces.numpy_fft.fft2(np.rot90(np.rot90(a)), [outsize[0], outsize[1]])
#     Fb = pyfftw.interfaces.numpy_fft.fft2(b, [outsize[0], outsize[1]])
#
#     xcorr_ab = pyfftw.interfaces.numpy_fft.ifft2(Fa * Fb)
#
#     return xcorr_ab

def xcorr2_fast(T, A):
    cross_corr = fftconvolve(np.flipud(np.fliplr(T)), A)

    return np.real(cross_corr)


def local_sum(A, m, n):
    B = np.pad(A, ((m, m), (n, n)), 'constant')
    s = np.cumsum(B, axis=0)
    c = s[m:(s.shape[0] - 1), :] - s[0:(s.shape[0] - m - 1), :]
    s = np.cumsum(c, axis=1)
    local_sum_A = s[:, n:s.shape[1] - 1] - s[:, 0:s.shape[1] - n - 1]
    return local_sum_A


def normxcorr2(b, a):
    # c = conv2(a,np.flipud(np.fliplr(b)))
    # a = conv2(a**2, np.ones(b.shape))
    # b = sum(b.flatten()**2)
    # c = c/np.sqrt(a*b)
    # return c

    b = b.astype(np.float)
    a = a.astype(np.float)
    xcorr_TA = xcorr2_fast(b, a)

    m = b.shape[0]
    n = b.shape[1]
    mn = m * n

    local_sum_A = local_sum(a, m, n)
    local_sum_A2 = local_sum(a * a, m, n)

    diff_local_sums = (local_sum_A2 - (local_sum_A ** 2) / mn)
    denom_A = np.sqrt(np.maximum(diff_local_sums, 0))

    denom_T = np.sqrt(mn - 1) * np.std(b)
    denom = np.dot(denom_T, denom_A)
    numerator = (xcorr_TA - local_sum_A * np.sum(b) / mn)

    C = np.zeros(numerator.shape)
    tol = np.sqrt(np.spacing(np.amax(np.abs(denom))))
    i_nonzero = denom > tol
    C[i_nonzero] = numerator[i_nonzero] / denom[i_nonzero]
    C[(np.abs(C) - 1) > np.sqrt(np.spacing(1))] = 0

    return C


######################################################
def deconvolveHE(I):
    I = I.astype(np.float32)
    h = I.shape[0]
    w = I.shape[1]
    c = I.shape[2]
    assert (c == 3), "An image must be RGB!"

    M = np.array([[0.644211, 0.716556, 0.266844], [0.092789, 0.954111, 0.283111]])
    M = np.vstack((M, np.cross(M[0, :], M[1, :])))
    M = M / np.transpose(np.tile(np.sqrt(np.sum(M ** 2, axis=1)), (3, 1)))
    I0 = 255.0

    J = np.reshape(I, (-1, 3))

    OD = -np.log((J + 1.0) / I0)

    C = np.transpose(np.linalg.solve(np.transpose(M), np.transpose(OD)))

    H = I0 * np.exp(np.dot(np.reshape(C[:, 0], (-1, 1)), -np.reshape(M[0, :], (1, -1))))
    H = np.reshape(H, (h, w, c))
    H = np.clip(H, 0, 255)
    H = H.astype(np.uint8)

    # E = I0 * np.exp(np.dot(np.reshape(C[:, 1], (-1, 1), order='F'), -np.reshape(M[1, :], (1, -1), order='F')))
    # E = np.reshape(E, (h, w, c), order='F')
    # E = np.clip(E, 0, 255)
    # E = E.astype(np.uint8)

    # Bg = I0 * np.exp(np.dot(np.reshape(C[:, 2], (-1, 1), order='F'), -np.reshape(M[2, :], (1, -1), order='F')))
    # Bg = np.reshape(Bg, (h, w, c), order='F')
    # Bg = np.clip(Bg, 0, 255)
    # Bg = Bg.astype(np.uint8)

    return H




#############################################################
def register_dapi2he(dapi, he, img):
    dapi = copy.copy(dapi)
    
    H = deconvolveHE(he)
    H = rgb2gray(H)
    H = 255 - H
    
    
    dapi[:,:,0] = 0 
    dapi[:,:,1] = 0 
    DAPI = np.uint8(np.zeros(he.shape))
    DAPI[:dapi.shape[0],:dapi.shape[1],:] = dapi
    
    new_img = np.uint8(np.zeros(he.shape))
    new_img[:img.shape[0],:img.shape[1],:] = img

    
    Transformation = registration_corr2_translation(DAPI[:,:,2],H)
    new_img = cv2.warpAffine(new_img, Transformation, (new_img.shape[1], new_img.shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return new_img  
#############################################################
def registration_corr2_translation(moving, fixed):
    scale = 4

    ##########################################################
    # leftover = int(moving.shape[0] % (2.0 ** scale))
    # if leftover != 0:
    #     moving = np.pad(moving, ((0, leftover), (0, 0)), 'constant', constant_values=0)
    #
    # leftover = int(moving.shape[1] % (2.0 ** scale))
    # if leftover != 0:
    #     moving = np.pad(moving, ((0, 0), (0, leftover)), 'constant', constant_values=0)
    #
    # leftover = int(fixed.shape[0] % (2.0 ** scale))
    # if leftover != 0:
    #     fixed = np.pad(fixed, ((0, leftover), (0, 0)), 'constant', constant_values=0)
    #
    # leftover = int(fixed.shape[1] % (2.0 ** scale))
    # if leftover != 0:
    #     fixed = np.pad(fixed, ((0, 0), (0, leftover)), 'constant', constant_values=0)
    ##########################################################

    moving_small = imresize(moving, 1.0 / (2.0 ** scale))
    fixed_small = imresize(fixed, 1.0 / (2.0 ** scale))

    if moving_small.shape[0] > fixed_small.shape[0]:
        fixed_small = np.pad(fixed_small,
                             ((0, moving_small.shape[0] - fixed_small.shape[0]),
                              (0, 0)),
                             'constant')
    if moving_small.shape[1] > fixed_small.shape[1]:
        fixed_small = np.pad(fixed_small,
                             ((0, 0),
                              (0, moving_small.shape[1] - fixed_small.shape[1])),
                             'constant')

    PCImage = normxcorr2(moving_small, fixed_small)
    ###############################################
    # ylim = 1000
    # xlim = 1000
    #
    # end_y = np.int(np.round(ylim/np.float(2**scale) + moving_small.shape[0] - 1))
    # end_x = np.int(np.round(xlim/np.float(2**scale) + moving_small.shape[1] - 1))
    #
    # PCImage = PCImage[0:end_y,0:end_x]

    ###############################################
    PCImage[PCImage < np.amax(PCImage) * 0.25] = 0

    L = laplace(PCImage)

    L[PCImage < np.amax(PCImage) * 0.3] = 0
    ypeak = np.unravel_index(L.argmin(), L.shape)[0]
    xpeak = np.unravel_index(L.argmin(), L.shape)[1]

    yoffSet = ypeak - moving_small.shape[0] + 1
    xoffSet = xpeak - moving_small.shape[1] + 1

    yoffSet = np.int(yoffSet * (2.0 ** scale))
    xoffSet = np.int(xoffSet * (2.0 ** scale))

    M_coarse = np.float32([[1, 0, xoffSet], [0, 1, yoffSet]])
    moving_coarse = cv2.warpAffine(moving, M_coarse, (moving.shape[1], moving.shape[0]))

    # fine registration

    sigma = 10
    kernel_size = 6 * sigma - 1
    template = cv2.GaussianBlur(moving_coarse, (kernel_size, kernel_size), 0)
    source = cv2.GaussianBlur(fixed, (kernel_size, kernel_size), 0)

    source_ycenter = np.int(np.round(source.shape[0] / 2.0))
    source_xcenter = np.int(np.round(source.shape[1] / 2.0))

    template_ycenter = np.int(np.round(template.shape[0] / 2.0))
    template_xcenter = np.int(np.round(template.shape[1] / 2.0))

    if source.shape[0] > 600 and source.shape[1] > 600:
        size_source = 600
    else:
        size_source = np.amin(np.array(source.shape))

    if template.shape[0] > 500 and template.shape[1] > 500:
        size_template = 500
    else:
        size_template = np.amin(np.array(template.shape))

    source = source[source_ycenter - size_source:source_ycenter + size_source,
             source_xcenter - size_source:source_xcenter + size_source]
    template = template[template_ycenter - size_template:template_ycenter + size_template,
               template_xcenter - size_template:template_xcenter + size_template]

    PCImage = normxcorr2(template, source)
    PCImage[PCImage < np.amax(PCImage) * 0.25] = 0
    L = laplace(PCImage)
    L[PCImage < np.amax(PCImage) * 0.3] = 0
    ypeak = np.unravel_index(L.argmin(), L.shape)[0]
    xpeak = np.unravel_index(L.argmin(), L.shape)[1]

    yoffSet = ypeak - template.shape[0] + 1
    xoffSet = xpeak - template.shape[1] + 1

    yoffSet = yoffSet - (size_source - size_template)
    xoffSet = xoffSet - (size_source - size_template)

    if np.abs(yoffSet) > 20 or np.abs(xoffSet) > 20:
        yoffSet = 0
        xoffSet = 0
        M = np.float32([[1, 0, xoffSet], [0, 1, yoffSet]])
    else:
        M = np.float32([[1, 0, xoffSet], [0, 1, yoffSet]])

    M[0, 2] += M_coarse[0, 2]
    M[1, 2] += M_coarse[1, 2]

    return M


#############################################################
def register_HE2FISH_translation(he, mask, fish):
    H = deconvolveHE(he)
    H = rgb2gray(H)
    H = 255 - H

    DAPI = fish[:, :, 2]

    Transformation = registration_corr2_translation(H, DAPI)
    new_he = cv2.warpAffine(he, Transformation, (he.shape[1], he.shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    new_mask = cv2.warpAffine(mask, Transformation, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return new_he, new_mask


#############################################################
def register_CK2FISH_translation(ck, fish):
    sigma = 5
    kernel_size = 6 * sigma - 1

    CK = cv2.GaussianBlur(ck[:, :, 2], (kernel_size, kernel_size), 0)

    DAPI = fish[:, :, 2]

    Transformation = registration_corr2_translation(CK, DAPI)

    new_ck = cv2.warpAffine(ck, Transformation, (ck.shape[1], ck.shape[0]))

    return new_ck


#############################################################
def skeleton(image):
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = np.squeeze(image)

    return skeletonize(image)


def cv2skeleton(img):
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = np.squeeze(img)

    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    if img.dtype == 'bool':
        img = img * 255.0
        img = img.astype(np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel


def find_skeleton3(img):
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = np.squeeze(img)

    skeleton = np.zeros(img.shape, np.uint8)
    eroded = np.zeros(img.shape, np.uint8)
    temp = np.zeros(img.shape, np.uint8)

    if img.dtype == 'bool':
        img = img * 255.0
        img = img.astype(np.uint8)

    _, thresh = cv2.threshold(img, 127, 255, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    iters = 0
    while (True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh  # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton, iters)


#############################################################

def imfill(im_th):
    if im_th.dtype == 'bool':
        im_th = im_th * 255.0
        im_th = im_th.astype(np.uint8)

    # Copy the thresholded image.
    im_th = np.lib.pad(im_th, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    im_out = im_out[1:h - 1, 1:w - 1]

    return im_out.astype(np.bool)
