from skimage.io import imread
from skimage.exposure import histogram,equalize_adapthist,equalize_hist
from skimage.filters import threshold_otsu,threshold_yen
import matplotlib.pyplot as plt
import numpy as np


# Leer imagen de apuntes
ruta = "imagenes/evaluacionP/evalproj-04.png"

im = imread(ruta,as_grey=True)


bin1 = im > threshold_otsu(im)
bin2 = im > threshold_yen(im)

eq1=equalize_hist(im)
eq2=equalize_adapthist(im)


f,ax=plt.subplots(1,3)
ax[0].imshow(bin1)
ax[1].imshow(bin2)
ax[2].imshow(im)
plt.show()

print("ho")

# Paso quitar lo que no sea texto o dibujos

# Pasar a escala de grises