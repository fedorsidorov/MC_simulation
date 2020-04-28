import numpy as np
from scipy import interpolate
import sys
import constants as c


# %%
def progress_bar(progress, total):
    barLength, status = 20, ''
    progress = float(progress) / float(total)

    if progress >= 1.:
        progress, status = 1, '\r\n'

    block = int(round(barLength * progress))

    text = '\r[{}] {:.0f}% {}'.format(
        '#' * block + '-' * (barLength - block), round(progress * 100, 0), status)

    sys.stdout.write(text)
    sys.stdout.flush()


# %% interpolation
def log_log_interp(xp, yp, kind='linear'):
    log_xp = np.log10(xp)
    log_yp = np.log10(yp)
    interp = interpolate.interp1d(log_xp, log_yp, kind=kind)
    def func(x): return np.power(10.0, interp(np.log10(x)))
    return func


def log_log_interp_2d(xp, yp, zp, kind='linear'):
    log_xp = np.log10(xp)
    log_yp = np.log10(yp)
    log_zp = np.log10(zp)
    interp = interpolate.interp2d(log_xp, log_yp, log_zp, kind=kind)
    def func(x, y): return np.power(10.0, interp(np.log10(x), np.log10(y)))
    return func


def lin_log_interp(xp, yp, kind='linear'):
    log_yp = np.log10(yp)
    interp = interpolate.interp1d(xp, log_yp, kind=kind)
    def func(x): return np.power(10.0, interp(x))
    return func


def log_lin_interp(xp, yp, kind='linear'):
    log_xp = np.log10(xp)
    interp = interpolate.interp1d(log_xp, yp, kind=kind)
    def func(x): return interp(np.log10(x))
    return func


# %% ELF utilities
def get_km_kp(E, hw):
    km = np.sqrt(2 * c.m / c.hbar ** 2) * (np.sqrt(E) - np.sqrt(E - hw))
    kp = np.sqrt(2 * c.m / c.hbar ** 2) * (np.sqrt(E) + np.sqrt(E - hw))
    return km, kp


# %% Monte-Carlo functions
def get_cumulated_array(array):
    result = np.zeros((len(array)))

    for i in range(len(array)):
        if np.all(array == 0):
            continue
        result[i] = np.sum(array[:i + 1])

    return result


def norm_2d_array(array, axis):
    result = np.zeros(np.shape(array))

    for i in range(np.shape(array)[axis]):
        if axis == 1:
            if np.sum(array[i, :]) != 0:
                result[i, :] = array[i, :] / np.sum(array[i, :])
                continue
        elif axis == 0:
            if np.sum(array[:, i]) != 0:
                result[:, i] = array[:, i] / np.sum(array[:, i])
                continue
        else:
            print('Specify axis - 0 ro 1')
            return -1

    return result
