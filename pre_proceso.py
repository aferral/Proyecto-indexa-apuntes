from skimage.filters import gaussian
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from skimage.morphology import binary_erosion, skeletonize, \
    remove_small_objects, remove_small_holes, watershed


def process(im):
    p1 = gaussian(im, sigma=2)
    p2 = gaussian(im, sigma=1)
    res = p1 - p2

    to_use = res

    # Hasta ahora lo mejor ha sido threshold_otsu(threshold_sauvola) siempre segmente texto de papel, pero a veces
    # viene entre medio de trozo o cambia cual es 0 cual es 1

    bin1 = to_use < threshold_otsu(threshold_niblack(to_use))

    bin1 = remove_small_holes(bin1)
    bin1 = remove_small_objects(bin1)

    return bin1