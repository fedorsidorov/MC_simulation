import importlib

import numpy as np

import indexes as ind
import constants as const

ind = importlib.reload(ind)
const = importlib.reload(const)


# %%
def get_e_id_DATA_ind_range(DATA, e_id):
    e_id_inds = np.where(DATA[:, ind.DATA_e_id_ind] == e_id)[0]
    beg_ind = e_id_inds[0]
    secondary_inds = np.where(DATA[:, ind.DATA_parent_e_id_ind] == e_id)[0]
    if len(secondary_inds) == 0:  # primary e created no 2ndaries
        end_ind = e_id_inds[-1]
    else:
        end_ind = secondary_inds[-1]  # there are secondary electrons
    return range(beg_ind, end_ind + 1)


def get_e_id_DATA(DATA, e_id):
    return DATA[get_e_id_DATA_ind_range(DATA, e_id), :]


def rotate_DATA(DATA, phi=2 * np.pi * np.random.random()):
    rot_mat = np.mat([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
    DATA[:, ind.DATA_x_ind:ind.DATA_E_dep_ind] = \
        np.dot(rot_mat, DATA[:, ind.DATA_x_ind:ind.DATA_E_dep_ind].transpose()).transpose()


def add_uniform_xy_shift_to_track(track_DATA, x_range, y_range):
    x_shift, y_shift = np.random.uniform(*x_range), np.random.uniform(*y_range)
    track_DATA[:, ind.DATA_xy_inds] += x_shift, y_shift


def add_gaussian_xy_shift_to_track(track_DATA, x_position, x_sigma, y_range):
    x_shift, y_shift = np.random.normal(loc=x_position, scale=x_sigma), np.random.uniform(*y_range)
    track_DATA[:, ind.DATA_xy_inds] += x_shift, y_shift


# def add_uniform_xy_shift_to_DATA(DATA, x_range, y_range):
#     n_e_prim = int(DATA[np.where(DATA[:, ind.DATA_parent_e_id_ind] == -1)][-1, 0] + 1)
#     for e_id in range(n_e_prim):
#         add_xy_shift_to_track(DATA, e_id, np.random.uniform(*x_range), np.random.uniform(*y_range))


# def get_daussian_xy_shift_to_DATA(DATA, r_beam, y_range):
#     n_e_prim = int(DATA[np.where(DATA[:, ind.DATA_parent_e_id_ind] == -1)][-1, 0] + 1)
#     for e_id in range(n_e_prim):
#         add_xy_shift_to_track(DATA, e_id, np.random.normal(loc=0, scale=r_beam), np.random.uniform(*y_range))


def get_n_electrons_2D(dose_uC_cm2, lx_nm, ly_nm):
    A_cm2 = lx_nm * ly_nm * 1e-14
    Q_C = dose_uC_cm2 * 1e-6 * A_cm2
    return int(np.round(Q_C / const.e_SI))

# def get_n_electrons_1D(dose_C_cm, ly_nm, y_borders_nm):
#     q_el_C = 1.6e-19
#     L_cm = (ly_nm + y_borders_nm * 2) * 1e-7
#     Q_C = dose_C_cm * L_cm
#     return int(np.round(Q_C / q_el_C))