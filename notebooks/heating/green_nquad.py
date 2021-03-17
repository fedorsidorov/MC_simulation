import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import nquad
import mcint
import random

from functions import MC_functions as mcf
import grid

grid = importlib.reload(grid)
mcf = importlib.reload(mcf)

# %%
k_1, k_2 = 0.002, 0.014  # W / cm / K
rho_1, rho_2 = 1.2, 2.2  # g / cm^3
Cp_1, Cp_2 = 1.5, 0.75  # J / g / K
alpha_1, alpha_2 = k_1 / (rho_1 * Cp_1), k_2 / (rho_2 * Cp_2)
lambda_12 = (k_1 * np.sqrt(alpha_2) - k_2 * np.sqrt(alpha_1)) / (k_1 * np.sqrt(alpha_2) + k_2 * np.sqrt(alpha_1))

L = 400e-7  # cm
n_terms = 5


def get_green_x(x, xp, t, tp, alpha):
    if t - tp < 0:
        print('green x error!')

    exp = np.exp(-(x - xp)**2 / (4 * alpha * (t - tp)))

    if exp > 1:
        print('green x exp > 1 error!')

    mult = 1 / np.sqrt(4 * np.pi * alpha * (t - tp))

    return mult * exp


def get_green_y(y, yp, t, tp, alpha):
    if t - tp < 0:
        print('green y error!')

    exp = np.exp(-(y - yp)**2 / (4 * alpha * (t - tp)))

    if exp > 1:
        print('green y exp > 1 error!')

    mult = 1 / np.sqrt(4 * np.pi * alpha * (t - tp))

    return mult * exp


def get_green_RR(z, zp, t, tp):

    if t - tp < 0:
        print('green RR error!')

    beg_mult = 1 / np.sqrt(4 * np.pi * alpha_1 * (t - tp))

    exp_1 = np.exp(-(z - zp)**2 / (4 * alpha_1 * (t - tp)))
    exp_2 = np.exp(-(z + zp)**2 / (4 * alpha_1 * (t - tp)))

    if exp_1 > 1 or exp_2 > 1:
        print('exp_12 RR error!')

    total_sum = 0

    for n in range(1, n_terms + 1):

        exp_3 = np.exp(-(z - zp + 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp)))
        exp_4 = np.exp(-(z + zp - 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp)))
        exp_5 = np.exp(-(z - zp - 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp)))
        exp_6 = np.exp(-(z + zp + 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp)))

        if exp_3 > 1 or exp_4 > 1 or exp_5 > 1 or exp_6 > 1:
            print('exp_3456 RR error!')

        total_sum += lambda_12**n * (exp_3 + exp_4 + exp_5 + exp_6)

    return beg_mult * (exp_1 + exp_2 + total_sum)


def get_green_RS(z, zp, t, tp):

    if t - tp < 0:
        print('green RS error!')

    beg_mult = (1 - lambda_12) / np.sqrt(4 * np.pi * alpha_2 * (t - tp))

    total_sum = 0

    for n in range(1, n_terms + 1):

        exp_1 = np.exp(-(np.sqrt(alpha_2 / alpha_1) * (-z + L + 2 * n * L) + (zp - L)) ** 2)
        exp_2 = np.exp(-(np.sqrt(alpha_2 / alpha_1) * (z + L + 2 * n * L) + (zp - L)) ** 2)

        if exp_1 > 1 or exp_2 > 1:
            print('exp_12 RS error!')

        total_sum += lambda_12 ** n * (exp_1 + exp_2)

    return beg_mult * total_sum


def get_green_z(z, zp, t, tp):

    if 0 <= z < L and 0 <= zp <= L:
        return get_green_RR(z, zp, t, tp)

    elif 0 <= z <= L < zp:
        return get_green_RS(z, zp, t, tp)

    else:
        print('green z error!')


def get_green_xyz(x, xp, y, yp, z, zp, t, tp):

    green_x = 0
    green_y = 0
    green_z = 0

    if 0 <= z < L and 0 <= zp <= L:
        green_x = get_green_x(x, xp, t, tp, alpha_1)
        green_y = get_green_x(y, yp, t, tp, alpha_1)
        green_z = get_green_RR(z, zp, t, tp)

    if 0 <= z <= L < zp:
        green_x = get_green_x(x, xp, t, tp, alpha_2)
        green_y = get_green_x(y, yp, t, tp, alpha_2)
        green_z = get_green_RS(z, zp, t, tp)

    return green_x * green_y * green_z


# %%
e = eV = 1.6e-19  # J

paper = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/heating/curves/dE_dx_chu.txt')
zz_cm, dE_dz_J_cm = paper[:, 0] * 1e-4, paper[:, 1] * 1000 * eV / 1e-4

# np.trapz(yy_chu, x=xx_chu)

lx = ly = 0.2e-4  # cm
lz = 21e-4
D = 10e-6  # C / cm^2
j = 250  # A / cm^2

t_exp = D / j
n_electrons = D * lx * ly / e
area = lx * ly


def get_g(xp, yp, zp, tp):
    if np.abs(xp) > lx / 2 or np.abs(yp) > ly / 2 or zp > lz or tp > t_exp:
        return 0

    now_dE_ds_J_cm = mcf.lin_lin_interp(zz_cm, dE_dz_J_cm)(zp)
    g = now_dE_ds_J_cm * n_electrons / area
    return g


# %%
plt.figure(dpi=300)

for coef in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    x_f, y_f, z_f, t_f = 0, 0, L * coef, t_exp * 0.5
    tt = np.linspace(0, t_f * 0.9, 100)
    gg = np.zeros(len(tt))

    for i in range(len(tt)):
        gg[i] = get_green_z(z_f, 0, t_f, tt[i])

    plt.plot(tt, gg, label=str(coef))

plt.legend()
plt.grid()
plt.show()


# %%
x_f, y_f, z_f, t_f = 0, 0, L * coef, t_exp * 0.5
# xp = 0
# yp = 0


def get_Y(xp, yp, zp, tp):
    return 1 / (rho_1 * Cp_1) * get_green_xyz(x_f, xp, y_f, yp, z_f, zp, t_f, tp) * get_g(xp, yp, zp, tp)


# %%
# integral, error =
nquad(get_Y, [
    [-lx/10, lx/10],
    [-ly/10, ly/10],
    [0, lz/10],
    [0, t_f * 0.5]
])

# T_final = mult * integral
