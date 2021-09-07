# %%
import numpy as np
import matplotlib.pyplot as plt


def snake_coord_1d(array, coord_min, coord_max):

    lower_el_inds = np.where(array < coord_min)[0]

    while len(lower_el_inds) > 0:
        array[lower_el_inds] += coord_max - coord_min
        lower_el_inds = np.where(array < coord_min)[0]

    upper_el_inds = np.where(array > coord_max)[0]

    while len(upper_el_inds):
        array[upper_el_inds] -= coord_max - coord_min
        upper_el_inds = np.where(array >= coord_max)[0]


def snake_coord(array, coord_ind, coord_min, coord_max):

    while True:
        lower_el_inds = np.where(array[:, coord_ind] < coord_min)[0]
        if len(lower_el_inds) == 0:
            break
        array[lower_el_inds, coord_ind] += coord_max - coord_min

    while True:
        upper_el_inds = np.where(array[:, coord_ind] >= coord_max)[0]
        if len(upper_el_inds) == 0:
            break
        array[upper_el_inds, coord_ind] -= coord_max - coord_min


def snake_array(array, x_ind, y_ind, z_ind, xyz_min, xyz_max):
    snake_coord(array, x_ind, xyz_min[0], xyz_max[0])
    snake_coord(array, y_ind, xyz_min[1], xyz_max[1])
    snake_coord(array, z_ind, xyz_min[2], xyz_max[2])
