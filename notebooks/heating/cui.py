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
Q = 20e-6 * 1e+4  # C / m^2

d = 1e-6  # m

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
# Rg_2 = 4.57e-5 / rho_2 * (E0 / 1e+3) ** 1.75

n_terms = 50


# %%
def get_lambda(ksi):
    return 0.6 + 6.21 * ksi - 12.4 * ksi**2 + 5.69 * ksi**3


def get_h(xp, yp, zp, tp, a, b, t_e):
    if np.abs(xp) > a / 2 or np.abs(yp) > b / 2 or zp > d or tp > t_e:
        return 0
    return E0 * Q * get_lambda(zp / Rg_1) / Rg_1 / t_e


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

        if expr_1 * (expr_2 + expr_3) < 0:
            print('get_g resist error!')

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


# %% MC integration - t
# nmc = 10000
#
# # a = b = 0.5e-6  # m
# t_e = 1e-6  # s
# tt = np.linspace(0.01, 4.01, 20) * 1e-6
# results = np.zeros(len(tt))
#
# # tt_e = np.array([0.5e-6, 1e-6, 1.5e-6, 3e-6])
# ab_arr = np.array([0.25e-6, 0.5e-6, 1e-6, 1.5e-6])
# T_total = np.zeros([len(ab_arr), len(tt)])
#
# for i, now_ab in enumerate(ab_arr):
#
#     print(now_ab)
#     a, b = now_ab, now_ab
#     progress_bar = tqdm(total=len(tt), position=0)
#
#     for j, now_t in enumerate(tt):
#
#         x_f, y_f, z_f, t_f = 0, 0, 0, now_t
#         domainsize = a * b * d * t_f
#
#         def integrand(xx):
#             xp = xx[0]
#             yp = xx[1]
        #     zp = xx[2]
        #     tp = xx[3]
        #     return get_Y(x_f, y_f, z_f, t_f, xp, yp, zp, tp, a, b, t_e)
        #
        # def sampler():
        #     while True:
        #         xp = random.uniform(-a / 2, a / 2)
        #         yp = random.uniform(-b / 2, b / 2)
        #         zp = random.uniform(0, d)
        #         tp = random.uniform(0, t_f)
        #         yield (xp, yp, zp, tp)
        #
        # T_total[i, j] = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)[0]
        #
        # progress_bar.update()

# %% 3a
# paper = np.loadtxt('notebooks/heating/curves/cui_3_a.txt')
#
# plt.figure(dpi=300)
#
# for i in range(len(ab_arr)):
#     plt.plot(tt * 1e+6, T_total[i, :], label='sim, a, b = ' + str(ab_arr[i] * 1e+6) + r' $\mu$m')
#
# plt.plot(paper[:, 0], paper[:, 1], '.', label='paper')
#
# plt.xlim(0, 2)
# plt.ylim(0, 300)
# plt.legend()
# plt.xlabel(r'time, $\mu$s')
# plt.ylabel(r'T, °C')
# plt.grid()
# # plt.show()
# plt.savefig('figure_3a.jpg')

# %% 3b
# paper = np.loadtxt('notebooks/heating/curves/cui_3_b.txt')
#
# plt.figure(dpi=300)
#
# for i in range(len(tt_e)):
#     plt.plot(tt * 1e+6, T_total[i, :], label='sim, t_exp = ' + str(tt_e[i] * 1e+6) + r' $\mu$s')
#
# plt.plot(paper[:, 0], paper[:, 1], '.', label='paper')
#
# plt.xlim(0, 4)
# plt.ylim(0, 150)
# plt.legend()
# plt.xlabel(r'time, $\mu$s')
# plt.ylabel(r'T, °C')
# plt.grid()
# # plt.show()
# plt.savefig('figure_3a.jpg')

# %% MC integration - z
nmc = 100000

a = b = 0.5e-6  # m
t_e = 1e-6  # s
zz = np.linspace(0, 1.5, 20) * 1e-6
results = np.zeros(len(zz))

progress_bar = tqdm(total=len(zz), position=0)

for i, now_z in enumerate(zz):

    x_f, y_f, z_f, t_f = 0, 0, now_z, t_e
    domainsize = a * b * d * t_f

    def integrand(xx):
        xp = xx[0]
        yp = xx[1]
        zp = xx[2]
        tp = xx[3]
        return get_Y(x_f, y_f, z_f, t_f, xp, yp, zp, tp, a, b, t_e)

    def sampler():
        while True:
            xp = random.uniform(-a / 2, a / 2)
            yp = random.uniform(-b / 2, b / 2)
            zp = random.uniform(0, d)
            tp = random.uniform(0, t_f)
            yield (xp, yp, zp, tp)

    results[i] = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)[0]

    progress_bar.update()

# %% print z
paper = np.loadtxt('notebooks/heating/curves/cui_2.txt')

plt.figure(dpi=300)

plt.plot(zz * 1e+6, results, 'o-', label='simulation')
plt.plot(paper[:, 0], paper[:, 1], 'o-', label='paper')
plt.plot(np.ones(10), np.linspace(0, 150, 10), '--')

plt.xlim(0, 1.5)
plt.ylim(0, 150)
plt.legend()
plt.xlabel(r'depth, $\mu$m')
plt.ylabel(r'T, °C')
plt.grid()
plt.show()
# plt.savefig('figure_2.jpg')

# %%
xx = np.arange(-a/2, a/2 + 1e-9, 50e-9)
zz = np.arange(0, d + 1e-9, 50e-9)

test_h = np.zeros([len(xx), len(zz)])
test_h_MC = np.zeros([len(xx), len(zz)])

for i, x in enumerate(xx):
    for j, z in enumerate(zz):
        test_h[i, j] = get_h(x, 0, z, 0, a, b, t_e)

# %%
plt.figure(dpi=300)
plt.imshow(test_h)
plt.show()



