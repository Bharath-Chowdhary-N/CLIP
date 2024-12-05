from astropy.nddata import Cutout2D
from copy import deepcopy
import numpy as np

def return_cutout(image_data, ra, dec, size, image_wcs):
    x,y = image_wcs.wcs_world2pix(ra, dec, 0)
    cutout_data = Cutout2D(image_data, (x,y), (size,size), wcs=image_wcs, mode='partial', fill_value=0).data
    return cutout_data

def normalize_image_simulated(input_array):
    working_array = deepcopy(input_array)
#     working_array = working_array/np.sum(working_array)
    high_percentile_value = np.percentile(working_array, 99)
    working_array[np.where(working_array>high_percentile_value)] = high_percentile_value
    working_array = working_array/np.max(working_array)
    return working_array