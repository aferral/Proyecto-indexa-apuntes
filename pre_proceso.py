from skimage.feature import peak_local_max, ORB, daisy, corner_harris, \
    corner_peaks, hog
from skimage.filters import gaussian
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from skimage.morphology import binary_erosion, skeletonize, \
    remove_small_objects, remove_small_holes, watershed
from skimage.filters import sobel
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.util import invert
from scipy import ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.draw import circle_perimeter,line
import os

from siempre_utiles import do_profile

def get_boxes(labels,n_labels):
    assert (len(labels.shape) == 2)
    boxes = []

    for i in range(n_labels):
        cords = np.where(labels == i)
        if len(cords[0]) == 0:
            continue
        x0,xf = cords[0].min(),cords[0].max()
        y0,yf = cords[1].min(),cords[1].max()
        box = (x0,xf,y0,yf)
        boxes.append(box)
    return boxes


def draw_boxes(labels, n_labels, fis=(20, 20)):
    assert (len(labels.shape) == 2)

    boxes = get_boxes(labels,n_labels)

    out = np.zeros((labels.shape[0], labels.shape[1], 3))
    out[:, :, 0] = labels

    for b in boxes: #box = (x0,xf,y0,yf)
        start = (b[0], b[2])
        end = (b[1], b[3])
        #         print(cords)
        #         print("rectangle {0} {1}".format(start,end))
        p0 = (start[0], start[1])
        p1 = (start[0], end[1])

        rr, cc = line(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]),
                      max(p0[1], p1[1]))
        out[rr, cc, 1] = 1

        p0 = (start[0], end[1])
        p1 = (end[0], end[1])

        rr, cc = line(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]),
                      max(p0[1], p1[1]))
        out[rr, cc, 1] = 1

        p0 = (end[0], end[1])
        p1 = (end[0], start[1])

        rr, cc = line(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]),
                      max(p0[1], p1[1]))
        out[rr, cc, 1] = 1

        p0 = (start[0], start[1])
        p1 = (end[0], start[1])

        rr, cc = line(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]),
                      max(p0[1], p1[1]))
        out[rr, cc, 1] = 1

    # rr, cc = rectangle(start[0], end[0], start[1], end[1])
    #         out[rr,cc,1] = 1

    #         plt.imshow(labels[start[0]:end[0],start[1]:end[1]])
    #         plt.show()
    #         break


    plt.figure(figsize=fis)
    plt.imshow(out)
    plt.show()



def process(im):

    to_use = sobel(im)


    bin1 = to_use < threshold_otsu(threshold_niblack(to_use))

    bin1 = remove_small_holes(bin1)
    bin1 = remove_small_objects(bin1)

    return invert(bin1)


def segmentacion(image):
    segments_fz = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    n_segmentos = segments_fz.max()
    # TODO FILTRAR BOXES INTERSECTADAS ???
    boxes = get_boxes(segments_fz, n_segmentos)

    return boxes

def procesar_dataset(imageList,feature_fun):

    dataset_features = []
    meta_data  = pd.DataFrame(columns=["x","y",'name'])


    for imgPath in imageList:
        if not os.path.isdir(imgPath):
            inputImg = imread(imgPath, as_grey=True)
            res = process(inputImg)
            data,meta = feature_fun(res,imgPath)
            dataset_features.append(data)
            meta_data=meta_data.append(meta,ignore_index=True)

    entire_data = np.concatenate(dataset_features,axis=0).reshape(-1,dataset_features[0].shape[1])
    return entire_data,meta_data

def features_region_HOG(img,imgName,show=False):
    cajas = segmentacion(img)

    cajasx0 = []
    cajasxf = []
    cajasy0 = []
    cajasyf = []
    vectores = []

    h = img.shape[0]
    w = img.shape[1]

    for box in cajas: #
        x0, xf, y0, yf = box

        if (abs(xf-x0) > ( h * 0.5) or abs(yf-y0) >  (0.5 *w) ):
            continue

        if (abs(xf-x0) < 16 or abs(yf-y0) < 16  ):
            continue

        region = img[x0:xf,y0:yf]

        if show:
            hog_v,hog_img = hog(region, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(3, 3), visualise=True,feature_vector=False)
            plt.imshow(hog_img)
            plt.show()
        else:
            hog_v = hog(region, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(3, 3), visualise=False,feature_vector=False)
        if len(hog_v) == 0:
            continue


        vector_acom = np.sum(hog_v,axis=(0,1,2,3))

        if vector_acom.sum() == 0:
            continue

        cajasx0.append(x0)
        cajasxf.append(xf)
        cajasy0.append(y0)
        cajasyf.append(yf)
        vectores.append(vector_acom / (vector_acom.sum()))

    features = np.concatenate(vectores, axis=0).reshape(-1,9)


    meta_data = pd.DataFrame(data={"x0" : cajasx0 ,
                                   "xf" : cajasxf,
                                   "y0" : cajasy0,
                                   "yf" : cajasyf,
                                   "name" : imgName})
    return features,meta_data

def features_ORB(img,imgName,show=False):

    descriptor_extractor = ORB(n_keypoints=1000)

    descriptor_extractor.detect_and_extract(img)
    keypoints = descriptor_extractor.keypoints
    features = descriptor_extractor.descriptors

    if show:
        new_rgb = np.zeros((img.shape[0],img.shape[1],3))
        new_rgb[:,:,0] = img

        for elem in keypoints:
            rr, cc = circle_perimeter(int(elem[0]),int(elem[1]), 5)
            rr = np.clip(rr,0,img.shape[0])
            cc = np.clip(cc, 0, img.shape[1])
            new_rgb[rr,cc,1] = 1
        print("corners {0}".format(features.shape[0]))
        plt.imshow(new_rgb)
        plt.show()

    meta_data = pd.DataFrame(data={"x" : keypoints[:,0] , "y" : keypoints[:,1], "name" : imgName})
    return features,meta_data


def features_harris(img,show=False):
    corners = corner_peaks(corner_harris(img))
    if show:
        new_rgb = np.zeros((img.shape[0], img.shape[1], 3))
        new_rgb[:, :, 0] = img
        for elem in corners:
            rr, cc = circle_perimeter(int(elem[0]),int(elem[1]), 5)
            rr = np.clip(rr,0,img.shape[0]-1)
            cc = np.clip(cc, 0, img.shape[1]-1)
            new_rgb[rr,cc,1] = 1
        plt.imshow(new_rgb)
        print("corners {0} ".format(corners.shape[0]))
        plt.show()

if __name__ == "__main__":
    ruta = "imagenes/evaluacionP/evalproj-04.png"

    im = imread(ruta, as_grey=True)

    res = process(im)
