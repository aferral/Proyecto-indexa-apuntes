from skimage.io import imread
import pandas as pd
from skimage.exposure import histogram,equalize_adapthist,equalize_hist
from skimage.filters import threshold_otsu,threshold_yen
import matplotlib.pyplot as plt
import numpy as np
from pre_proceso import process, segmentacion, features_ORB, features_harris, \
    procesar_dataset, features_region_HOG
from skimage.morphology import watershed, skeletonize
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from siempre_utiles import string_fecha
from collections import Counter
from skimage.feature import plot_matches, match_descriptors

def busqueda_densa(query_desc,query_meta,dataset,meta_data):
    nbrs = NearestNeighbors(n_neighbors=1).fit(dataset)
    distances, indices = nbrs.kneighbors(query_desc)

    # Con los vecinos mas cercanos encuentra la imagen con mas matches
    imagenes_cercanas = [meta_data.iloc[ind[0]]['name']for ind in indices]
    imagen_m_cercana = Counter(imagenes_cercanas).most_common()[1][0]

    # Ahora buscar solamente en la imagen mas cercana
    selected = meta_data['name'] == imagen_m_cercana
    filtrado = meta_data[selected]
    start_index = filtrado.index[0]
    end_index = filtrado.index[-1] + 1 #notacion de numpy requiere uno mas arriba
    dataset_filtrado = dataset[start_index:end_index]

    nbrs = NearestNeighbors(n_neighbors=1).fit(dataset_filtrado)
    distances_img, indices_img = nbrs.kneighbors(query_desc)



    imQ = imread(query_meta.iloc[0]['name'],as_grey=True)
    imb = imread(imagen_m_cercana,as_grey=True)
    key_q = np.vstack([q_meta['x'],q_meta['y']]).T
    key_b = np.vstack([filtrado['x'], filtrado['y']]).T
    n_q  = key_q.shape[0]

    # matches = match_descriptors(query_desc,dataset_filtrado)
    matches = np.hstack([np.arange(n_q).reshape(n_q,1),indices_img])

    fig, ax = plt.subplots(nrows=1, ncols=1)

    plot_matches(ax, imQ, imb, key_q, key_b, matches,
                 keypoints_color='k', matches_color=None, only_matches=False)
    plt.show()

    # TODO PESAR POR TF IDF

    print("hi")

def busqueda_box(query_desc, query_meta, dataset, meta_data):
    nbrs = NearestNeighbors(n_neighbors=1).fit(dataset)
    distances, indices = nbrs.kneighbors(query_desc)

    for ind in indices:
        x0=meta_data.iloc[ind]['x0']
        xf=meta_data.iloc[ind]['xf']
        y0=meta_data.iloc[ind]['y0']
        yf=meta_data.iloc[ind]['yf']

        fig, ax = plt.subplots(nrows=1, ncols=2)
        imQ = imread(query_meta.iloc[0]['name'], as_grey=True)
        imb = imread(meta_data.iloc[ind]['name'], as_grey=True)

        ax[0].imshow(imQ)
        ax[0].imshow(imb[x0:xf,y0:yf])
        plt.show()

    # TODO PESAR POR TF IDF

    print("hi")

    pass


carpetaImagenes = "imagenes/evaluacionP/"
dataset_a_recuperar = "HOG_window_1_2017_12_05_20-03-09.pkl" #"HOG_windows_1_2017_12_05_15-37-18.pkl"
feature_fun = features_region_HOG

imageList = list(map(lambda img : os.path.join(carpetaImagenes,img),os.listdir(carpetaImagenes)))


if dataset_a_recuperar is None:
    descriptores,meta_data = procesar_dataset(imageList,feature_fun)

    outName = 'HOG_window_1_{0}.pkl'.format(string_fecha())
    with open(outName,'wb') as f:
        pickle.dump((descriptores,meta_data),f)
    print("Dataset y meta_data guardados en {0}".format(outName))
else:

    # Cargar dataset
    with open(dataset_a_recuperar,'rb') as f:
        descriptores,meta_data = pickle.load(f)

    path = "imagenes/querys/van.png"
    query = imread(path,as_grey=True)
    query = threshold_otsu(query) > query # como era de binario ?? coinciden"??
    query_desc,q_meta =  feature_fun(query,path,show=True)

    busqueda_box(query_desc,q_meta, descriptores,meta_data)

