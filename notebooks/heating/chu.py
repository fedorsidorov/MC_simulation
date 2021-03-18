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
k_1, k_2 = 0.2, 1.4  # W / m / K
rho_1, rho_2 = 1200, 2200  # kg / cm^3
Cp_1, Cp_2 = 1500, 750  # J / kg / K
alpha_1, alpha_2 = k_1 / (rho_1 * Cp_1), k_2 / (rho_2 * Cp_2)
lambda_12 = (k_1 * np.sqrt(alpha_2) - k_2 * np.sqrt(alpha_1)) / (k_1 * np.sqrt(alpha_2) + k_2 * np.sqrt(alpha_1))

L = 400e-9  # cm
n_terms = 20


def get_green_x(x, xp, t, tp, alpha):
    exp = np.exp(-(x - xp)**2 / (4 * alpha * (t - tp)))
    mult = 1 / np.sqrt(4 * np.pi * alpha * (t - tp))
    return mult * exp


def get_green_y(y, yp, t, tp, alpha):
    exp = np.exp(-(y - yp)**2 / (4 * alpha * (t - tp)))
    mult = 1 / np.sqrt(4 * np.pi * alpha * (t - tp))
    return mult * exp


def get_green_RR(z, zp, t, tp):
    beg_mult = 1 / np.sqrt(4 * np.pi * alpha_1 * (t - tp))
    term_1 = np.exp(-(z - zp)**2 / (4 * alpha_1 * (t - tp)))
    term_2 = np.exp(-(z + zp)**2 / (4 * alpha_1 * (t - tp)))

    total_sum = 0

    for n in range(1, n_terms + 1):
        total_sum += lambda_12 ** n * (
            np.exp(-(z - zp + 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp))) +
            np.exp(-(z + zp - 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp))) +
            np.exp(-(z - zp - 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp))) +
            np.exp(-(z + zp + 2 * n * L) ** 2 / (4 * alpha_1 * (t - tp)))
        )

    return beg_mult * (term_1 + term_2 + total_sum)


def get_green_RS(z, zp, t, tp):
    beg_mult = (1 - lambda_12) / np.sqrt(4 * np.pi * alpha_2 * (t - tp))
    total_sum = 0

    for n in range(n_terms):
        total_sum += lambda_12 ** n * (
                np.exp(-(np.sqrt(alpha_2 / alpha_1) * (-z + L + 2 * n * L) + (zp - L)) ** 2) +
                np.exp(-(np.sqrt(alpha_2 / alpha_1) * (z + L + 2 * n * L) + (zp - L)) ** 2)
        )

    return beg_mult * total_sum


def get_green_z(z, zp, t, tp):

    if 0 <= z <= L and 0 <= zp <= L:
        return get_green_RR(z, zp, t, tp)

    elif 0 <= z <= L < zp:
        return get_green_RS(z, zp, t, tp)


def get_green_xyz(x, xp, y, yp, z, zp, t, tp):

    green_x = 0
    green_y = 0
    green_z = 0

    if 0 <= z <= L and 0 <= zp <= L:
        green_x = get_green_x(x, xp, t, tp, alpha_1)
        green_y = get_green_x(y, yp, t, tp, alpha_1)
        green_z = get_green_RR(z, zp, t, tp)

    elif 0 <= z <= L < zp:
        green_x = get_green_x(x, xp, t, tp, alpha_2)
        green_y = get_green_x(y, yp, t, tp, alpha_2)
        green_z = get_green_RS(z, zp, t, tp)

    return green_x * green_y * green_z


# %%
e = eV = 1.6e-19  # J

paper = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/notebooks/heating/curves/dE_dx_chu.txt')
zz_m, dE_dz_J_m = paper[:, 0] * 1e-6, paper[:, 1] * 1000 * eV / 1e-6

# np.trapz(yy_chu, x=xx_chu)

a = b = 0.2e-6  # m
Rg = 21e-6
D = 10e-6 * 1e+4  # C / m^2
j = 250 * 1e+4  # A / m^2
t_exp = D / j
n_electrons = D * a * b / e
area = a * b


def get_g(xp, yp, zp, tp):
    if np.abs(xp) > a / 2 or np.abs(yp) > b / 2 or zp > Rg or tp > t_exp:
        return 0

    now_dE_ds_J_cm = mcf.lin_lin_interp(zz_m, dE_dz_J_m)(zp)
    g = now_dE_ds_J_cm * n_electrons / area

    return g * 1e+7


# %% MC integration - t
nmc = 100000

coefs = [0.01, 0.2, 0.4, 0.6, 0.8, 1.01, 1.2, 1.4, 1.6, 1.8, 2.0]
results = np.zeros(len(coefs))

progress_bar = tqdm(total=len(coefs), position=0)

for i, coef in enumerate(coefs):

    x_f, y_f, z_f, t_f = 0, 0, 0, t_exp * coef
    domainsize = a * b * Rg * t_f

    def integrand(xx):
        xp = xx[0]
        yp = xx[1]
        zp = xx[2]
        tp = xx[3]

        return get_green_xyz(x_f, xp, y_f, yp, z_f, zp, t_f, tp) * get_g(xp, yp, zp, tp)


    def sampler():
        while True:
            xp = random.uniform(-a / 2, a / 2)
            yp = random.uniform(-b / 2, b / 2)
            zp = random.uniform(0, Rg)
            tp = random.uniform(0, t_f)
            yield (xp, yp, zp, tp)

    results[i] = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)[0]

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.plot(coefs, results)
plt.show()

# %% MC integration - z
nmc = 100000

coefs = [0.01, 0.2, 0.4, 0.6, 0.8, 1.01, 1.2, 1.4, 1.6, 1.8, 2.0]
results = np.zeros(len(coefs))

progress_bar = tqdm(total=len(coefs), position=0)

for i, coef in enumerate(coefs):

    x_f, y_f, z_f, t_f = 0, 0, L * coef, t_exp
    domainsize = a * b * Rg * t_f

    def integrand(xx):
        xp = xx[0]
        yp = xx[1]
        zp = xx[2]
        tp = xx[3]

        return get_green_xyz(x_f, xp, y_f, yp, z_f, zp, t_f, tp) * get_g(xp, yp, zp, tp)


    def sampler():
        while True:
            xp = random.uniform(-a / 2, a / 2)
            yp = random.uniform(-b / 2, b / 2)
            zp = random.uniform(0, Rg)
            tp = random.uniform(0, t_f)
            yield (xp, yp, zp, tp)

    results[i] = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)[0]

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.plot(coefs, results / Cp_1 / rho_1)
plt.show()


