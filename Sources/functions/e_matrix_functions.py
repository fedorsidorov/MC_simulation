import importlib
import numpy as np
import indexes as ind
import constants as const
from functions import MC_functions as mcf

ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
const = importlib.reload(const)


# %%
def get_e_id_e_DATA_ind_range(e_DATA, n_primaries_in_file, e_id):
    e_id_inds = np.where(e_DATA[:, ind.e_DATA_e_id_ind] == e_id)[0]

    if len(e_id_inds) == 0:
        return range(0, 0)

    if e_id == n_primaries_in_file - 1:
        return range(e_id_inds[0], len(e_DATA))

    next_e_id_inds = np.where(e_DATA[:, ind.e_DATA_e_id_ind] == e_id + 1)[0]

    while len(next_e_id_inds) == 0:
        e_id += 1

        if e_id == n_primaries_in_file - 1:
            return range(e_id_inds[0], len(e_DATA))
        else:
            next_e_id_inds = np.where(e_DATA[:, ind.e_DATA_e_id_ind] == e_id + 1)[0]

    return range(e_id_inds[0], next_e_id_inds[0])


def get_e_id_e_DATA(e_DATA, n_primaries_in_file, e_id):
    return e_DATA[get_e_id_e_DATA_ind_range(e_DATA, n_primaries_in_file, e_id), :]


def rotate_DATA(e_DATA, phi=2 * np.pi * np.random.random()):
    rot_mat = np.mat([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
    e_DATA[:, ind.e_DATA_x_ind:ind.e_DATA_z_ind] = \
        np.dot(rot_mat, e_DATA[:, ind.e_DATA_x_ind:ind.e_DATA_z_ind].transpose()).transpose()


def add_uniform_xy_shift_to_e_DATA(e_DATA, x_range, y_range):
    x_shift, y_shift = np.random.uniform(*x_range), np.random.uniform(*y_range)
    e_DATA[:, ind.e_DATA_xy_inds] += x_shift, y_shift


# def add_gaussian_xy_shift_to_track(track_DATA, x_position, x_sigma, y_range):
#     x_shift, y_shift = np.random.normal(loc=x_position, scale=x_sigma), np.random.uniform(*y_range)
#     track_DATA[:, ind.e_DATA_xy_inds] += x_shift, y_shift


def add_individual_uniform_xy_shifts_to_e_DATA(e_DATA, n_primaries_in_file, x_range, y_range):

    for e_id in range(n_primaries_in_file):
        now_e_DATA_range = get_e_id_e_DATA_ind_range(e_DATA, n_primaries_in_file, e_id)

        if len(now_e_DATA_range) > 0:
            x_shift, y_shift = np.random.uniform(*x_range), np.random.uniform(*y_range)
            e_DATA[now_e_DATA_range, ind.e_DATA_x_ind] += x_shift
            e_DATA[now_e_DATA_range, ind.e_DATA_y_ind] += y_shift


def delete_snaked_vacuum_events(e_DATA, xx, zz_vac):

    def get_z_vac(x):
        if x > np.max(xx):
            return zz_vac[-1]
        elif x < np.min(xx):
            return zz_vac[0]
        else:
            return mcf.lin_lin_interp(xx, zz_vac)(x)

    delete_inds = []

    for i, line in enumerate(e_DATA):
        now_x, now_z = line[ind.e_DATA_x_ind], line[ind.e_DATA_z_ind]
        if now_z < get_z_vac(now_x):
            delete_inds.append(i)

    return np.delete(e_DATA, delete_inds, axis=0)


def get_n_electrons_2D(dose_uC_cm2, lx_nm, ly_nm):
    A_cm2 = lx_nm * ly_nm * 1e-14
    Q_C = dose_uC_cm2 * 1e-6 * A_cm2
    return int(np.round(Q_C / const.e_SI))


def get_n_electrons_1D(dose_pC_cm, ly_nm):
    L_cm = ly_nm * 1e-7
    Q_C = dose_pC_cm * 1e-12 * L_cm
    return int(np.round(Q_C / const.e_SI))
