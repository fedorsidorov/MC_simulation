import importlib
import numpy as np
import indexes as ind
import constants as const
from functions import MC_functions as mcf

ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
const = importlib.reload(const)


# %%
def get_e_id_e_DATA_simple(e_DATA, n_primaries_in_file, e_id):

    if len(np.where(e_DATA[:, 0] == e_id)[0]) == 0:
        return None

    ind_now_prim_e = np.where(e_DATA[:, 0] == e_id)[0][0]

    ind_next_prim_e = len(e_DATA)

    where = np.where(np.logical_and(
                e_DATA[:, 0] > e_id,
                e_DATA[:, 0] < n_primaries_in_file
            ))[0]

    if e_id < n_primaries_in_file - 1 and len(where) != 0:
        ind_next_prim_e = where[0]

    return e_DATA[ind_now_prim_e:ind_next_prim_e, :]


def rotate_DATA(e_DATA, phi=2 * np.pi * np.random.random()):
    rot_mat = np.mat([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
    e_DATA[:, [ind.e_DATA_x_ind, ind.e_DATA_y_ind]] =\
        np.dot(rot_mat, e_DATA[:, [ind.e_DATA_x_ind, ind.e_DATA_y_ind]].transpose()).transpose()


def add_uniform_xy_shift_to_e_DATA(e_DATA, x_range, y_range):
    x_shift, y_shift = np.random.uniform(*x_range), np.random.uniform(*y_range)
    e_DATA[:, ind.e_DATA_xy_inds] += x_shift, y_shift


def add_gaussian_x_shift_to_e_DATA(e_DATA, sigma):
    x_shift = np.random.normal(loc=0, scale=sigma)
    e_DATA[:, ind.e_DATA_x_ind] += x_shift


def add_gaussian_xy_shift_to_e_DATA(e_DATA, x_position, x_sigma, y_range):
    x_shift, y_shift = np.random.normal(loc=x_position, scale=x_sigma), np.random.uniform(*y_range)
    e_DATA[:, ind.e_DATA_xy_inds] += x_shift, y_shift


def delete_snaked_vacuum_events(e_DATA, xx_vac, zz_vac):

    def get_z_vac(x):
        return mcf.lin_lin_interp(xx_vac, zz_vac)(x)

    delete_inds = []

    for i, line in enumerate(e_DATA):
        now_x, now_z = line[ind.e_DATA_x_ind], line[ind.e_DATA_z_ind]
        if now_z < get_z_vac(now_x):
            delete_inds.append(i)

    return np.delete(e_DATA, delete_inds, axis=0)
