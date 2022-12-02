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
e = eV = 1.6e-19

# exposure parameters
E0 = 20e+3
Q_C_cm2 = 0.87e-6
Q_C_m2 = Q_C_cm2 * 1e+4  # C / m^2

# d = 0.9e-6  # m
d = 0.5e-6  # m

# 1 - PMMA
D_1 = 0.2  # W / m / K
rho_1 = 1200  # kg / m^3
Cv_1 = 1500  # J / kg / K
k_1 = D_1 / (Cv_1 * rho_1)
Rg_1 = 4.57e-5 / rho_1 * (E0 / 1e+3) ** 1.75

# 2 - simple_Si_MC
D_2 = 150  # W / m / K
rho_2 = 2330  # kg / m^3
Cv_2 = 700  # J / kg / K
k_2 = D_2 / (Cv_2 * rho_2)

n_terms = 50

# %%
E_dep_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/notebooks/heating/heating_5000_files.npy')

lx_m, ly_m, lz_m = 5e+3 * 1e-9, 10e+3 * 1e-9, 6e+3 * 1e-9
n_files, n_primaries_in_file = 5000, 100
bin_size_m = 50 * 1e-9

x_bins_m = np.arange(-lx_m/2, lx_m/2 + 1e-9, bin_size_m)
z_bins_m = np.arange(0, lz_m + 1e-9, bin_size_m)

x_centers_m = (x_bins_m[:-1] + x_bins_m[1:]) / 2
z_centers_m = (z_bins_m[:-1] + z_bins_m[1:]) / 2

# n_e_1s = 0.85e-9 * lx_m * 100 * ly_m * 100 / e
# E_dep_1s = E_dep_matrix / n_files / n_primaries_in_file * n_e_1s
# E_dep_1s_J = E_dep_1s * eV
# E_dep_1s_J_m3 = E_dep_1s_J / bin_size_m**2 / ly_m

n_e_portion = 3
E_dep_portion = E_dep_1s = E_dep_matrix / n_files / n_primaries_in_file * n_e_portion


# %%
def get_lambda(ksi):
    return 0.6 + 6.21 * ksi - 12.4 * ksi**2 + 5.69 * ksi**3


def get_h(xp, yp, zp, tp, a, b, t_e):
    if np.abs(xp) > a / 2 or np.abs(yp) > b / 2 or zp > d or tp > t_e:
        return 0
    return E0 * Q_C_m2 * get_lambda(zp / Rg_1) / Rg_1 / t_e


def get_h_MC(xp, zp, tp, t_e):
    if tp > t_e:
        return 0

    x_pos = np.argmin(np.abs(x_centers_m - xp))
    z_pos = np.argmin(np.abs(z_centers_m - zp))
    return E_dep_portion[x_pos, z_pos]


h_easy = get_h(0, 0, 0, 0, 1, 1, 1000)
h_MC = get_h_MC(0, 0, 0, 1000)


# %%
def get_f(x, y, z, t, xp, yp, zp, tp):
    if zp <= d:
        k = k_1
    else:
        k = k_2

    expr_1 = 1 / (4 * np.pi * k * (t - tp))
    expr_2 = np.exp(-((x - xp)**2 + (y - yp)**2) / (4 * k * (t - tp)))

    if expr_1 * expr_2 < 0:
        print('get_f error!')

    return expr_1 * expr_2


def get_g(z, t, zp, tp):
    K = np.sqrt(k_1 / k_2)
    Kp = 1 / K
    sigma = D_2 / D_1 * np.sqrt(k_1 / k_2)
    alpha = (sigma + 1) / (sigma - 1)
    theta = D_1 / D_2 * k_2 / k_1
    eta = 2 * sigma * K * theta / (1 + sigma)
    beta = (1 + alpha) / sigma

    if z <= d:
        expr_1 = 1 / (2 * np.sqrt(np.pi * k_1 * (t - tp)))
        expr_2 = np.exp(-(z - zp) ** 2 / (4 * k_1 * (t - tp))) +\
            np.exp(-(z + zp) ** 2 / (4 * k_1 * (t - tp))) +\
            2 * sigma * K / (1 + sigma) * np.exp(-(d + z + K * (d - zp)) ** 2 / (4 * k_1 * (t - tp))) +\
            2 * sigma * K / (1 + sigma) * np.exp(-(d - z + K * (d - zp)) ** 2 / (4 * k_1 * (t - tp)))

        expr_3 = 0
        for n in range(1, n_terms + 1):
            expr_3 += 2 * sigma * K / (1 + sigma) *\
                      (-alpha) ** n * np.exp(-((2 * n + 1) * d - z + K * (d - zp)) / (4 * k_1 * (t - tp)))

            expr_3 += (-alpha) ** n * np.exp(-(z + zp + 2 * n * d) ** 2 / (4 * k_1 * (t - tp)))
            expr_3 += (-1) ** n * alpha ** (n - 1) * np.exp(-(z - zp + 2 * n * d) ** 2 / (4 * k_1 * (t - tp)))
            expr_3 += (-1) ** n * alpha ** (n - 1) * np.exp(-(-z - zp + 2 * n * d) ** 2 / (4 * k_1 * (t - tp)))
            expr_3 += (-alpha) ** n * np.exp(-(-z + zp + 2 * n * d) ** 2 / (4 * k_1 * (t - tp)))

        if expr_2 + expr_3 < 0:
            return 0

        # if expr_1 * (expr_2 + expr_3) < 0:
            # print('get_g resist error!', expr_3)

        return expr_1 * (expr_2 + expr_3)

    else:
        expr_1 = 1 / (2 * np.sqrt(np.pi * k_2 * (t - tp)))
        expr_2 = (2 - eta) * np.exp(-(z - zp) ** 2 / (4 * k_2 * (t - tp)))

        expr_3 = 0
        for n in range(1, n_terms + 1):
            expr_3 -= eta * (1 + 1 / alpha) * (-alpha) ** n *\
                      np.exp(-(-z + zp + 2 * n * Kp * d) / (4 * k_2 * (t - tp)))

            expr_3 += beta * (-alpha) ** (n - 1) *\
                np.exp(-(z - d - Kp * (zp + (2 * n - 1) * d)) ** 2 / (4 * k_2 * (t - tp)))
            expr_3 -= beta * (-alpha) ** (n - 1) * \
                np.exp(-(z - d - Kp * (zp - (2 * n + 1) * d)) ** 2 / (4 * k_2 * (t - tp)))

        if expr_1 * (expr_2 + expr_3) < 0:
            print('get_g silicon error!')

        return expr_1 * (expr_2 + expr_3)


def get_G(x, y, z, t, xp, yp, zp, tp):
    return get_f(x, y, z, t, xp, yp, zp, tp) * get_g(z, t, zp, tp)


def get_Y(x, y, z, t, xp, yp, zp, tp, a, b, t_e):
    if z <= d:
        return 1 / (Cv_1 * rho_1) * get_G(x, y, z, t, xp, yp, zp, tp) * get_h(xp, yp, zp, tp,  a, b, t_e)
    else:
        return 1 / (Cv_2 * rho_2) * get_G(x, y, z, t, xp, yp, zp, tp) * get_h(xp, yp, zp, tp,  a, b, t_e)


def get_Y_MC(x, y, z, t, xp, yp, zp, tp, t_e):
    if z <= d:
        return 1 / (Cv_1 * rho_1) * get_G(x, y, z, t, xp, yp, zp, tp) * get_h_MC(xp, zp, tp, t_e)
    else:
        return 1 / (Cv_2 * rho_2) * get_G(x, y, z, t, xp, yp, zp, tp) * get_h_MC(xp, zp, tp, t_e)


# %% MC integration - t
nmc = 10000
t_exp = 1.64e-9  # s

tt = np.linspace(0.01, 2.01, 50) * 1e-9
results = np.zeros(len(tt))
errors = np.zeros(len(tt))

progress_bar = tqdm(total=len(tt), position=0)

for i, now_t in enumerate(tt):

    x_f, y_f, z_f, t_f = 0, 0, 0, now_t
    domainsize = lx_m * ly_m * lz_m * t_f

    def integrand(xx):
        xp = xx[0]
        yp = xx[1]
        zp = xx[2]
        tp = xx[3]
        # return get_Y(x_f, y_f, z_f, t_f, xp, yp, zp, tp, lx_m, ly_m, t_exp)
        return get_Y_MC(x_f, y_f, z_f, t_f, xp, yp, zp, tp, t_exp)

    def sampler():
        while True:
            xp = random.uniform(-lx_m / 2, lx_m / 2)
            yp = random.uniform(-ly_m / 2, ly_m / 2)
            zp = random.uniform(0, lz_m)
            tp = random.uniform(0, t_f)
            yield (xp, yp, zp, tp)

    results[i], errors[i] = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)

    progress_bar.update()


# %%
plt.figure(dpi=300)
plt.plot(tt, results, 'o-')
plt.plot(tt, results + errors, '^--')
plt.plot(tt, results - errors, 'v--')

# plt.title('900 nm PMMA layer, 0.85 nA/cm$^2$, t$_{exp}$=1000 s')
plt.title('900 nm PMMA layer, 0.85 nA/cm$^2$')
plt.xlabel('t, s')
plt.ylabel(r'$\Delta$T, °C')

plt.grid()
# plt.savefig('DEBER_heating.jpg')
plt.show()

# %%
xx = x_centers_m
zz = z_centers_m

test_h = np.zeros([len(xx), len(zz)])
test_h_MC = np.zeros([len(xx), len(zz)])

for i, x in enumerate(xx):
    for j, z in enumerate(zz):
        test_h[i, j] = get_h(x, 0, z, 0, lx_m, ly_m, t_exp)
        test_h_MC[i, j] = get_h_MC(x, z, 0, t_exp)

# %%
print('Gruen dE / MC dE =', np.sum(test_h) / np.sum(test_h_MC[:, :18]))

# %%
plt.figure(dpi=300)
plt.imshow(test_h)
plt.show()

