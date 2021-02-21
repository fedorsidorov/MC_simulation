import numpy as np
from scipy import interpolate

import importlib

import constants as const
import grid

const = importlib.reload(const)
grid = importlib.reload(grid)


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


def lin_lin_interp(xp, yp, kind='linear'):
    interp = interpolate.interp1d(xp, yp, kind=kind)
    def func(x): return interp(x)
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
    km = np.sqrt(2 * const.m / const.hbar ** 2) * (np.sqrt(E) - np.sqrt(E - hw))
    kp = np.sqrt(2 * const.m / const.hbar ** 2) * (np.sqrt(E) + np.sqrt(E - hw))
    return km, kp


# %% Monte-Carlo functions
def get_cumulated_array(array, x):
    result = np.zeros((len(array)))

    for i in range(len(array)):
        if np.all(array == 0):
            continue
        result[i] = np.trapz(array[:i + 1], x=x[:i + 1])

    return result


def norm_2d_array(array):
    result = np.zeros(np.shape(array))

    for i in range(len(array)):
        if np.sum(array[i, :]) != 0:
            result[i, :] = array[i, :] / np.sum(array[i, :])

    return result


def get_IEMFP_from_DEIMFP(DIEMFP):
    IEMFP = np.zeros(grid.EE)
    for i, _ in enumerate(grid.EE):
        IEMFP[i, :] = np.trapz(DIEMFP[i, :] * 2 * np.pi * np.sin(grid.THETA_rad), x=grid.THETA_rad)

    return IEMFP


def get_IIMFP_from_DIIMFP(DIIMFP):
    IIMFP = np.zeros(grid.EE)
    for i, E in enumerate(grid.EE):
        inds = np.where(grid.EE < E)
        IIMFP[i, :] = np.trapz(DIIMFP[i, inds], x=grid.EE[inds])

    return IIMFP
