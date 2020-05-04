# %%
import numpy as np
import matplotlib.pyplot as plt


def snake_coord(arr, coord_ind, coord_min, coord_max):

    while True:
        lower_el_inds = np.where(arr[:, coord_ind] < coord_min)[0]
        if len(lower_el_inds) == 0:
            break
        arr[lower_el_inds, coord_ind] += coord_max - coord_min

    while True:
        upper_el_inds = np.where(arr[:, coord_ind] >= coord_max)[0]
        if len(upper_el_inds) == 0:
            break
        arr[upper_el_inds, coord_ind] -= coord_max - coord_min


def snake_array(array, x_ind, y_ind, z_ind, xyz_min, xyz_max):
    snake_coord(array, x_ind, xyz_min[0], xyz_max[0])
    snake_coord(array, y_ind, xyz_min[1], xyz_max[1])
    snake_coord(array, z_ind, xyz_min[2], xyz_max[2])
