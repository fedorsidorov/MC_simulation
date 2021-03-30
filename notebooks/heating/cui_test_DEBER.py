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
# Q_uC_cm2 = 20
Q_uC_cm2 = 0.02
Q = Q_uC_cm2 * 1e-6 * 1e+4  # C / m^2
# Q = 1.2e-9 * 1e+4  # C / m^2

d = 1e-6  # m

# 1 - PMMA
D_1 = 0.2  # W / m / K
rho_1 = 1200  # kg / m^3
Cv_1 = 1500  # J / kg / K
k_1 = D_1 / (Cv_1 * rho_1)
Rg_1 = 4.57e-5 / rho_1 * (E0 / 1e+3) ** 1.75

# 2 - Si
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

        return expr_1 * (expr_2 + expr_3)


def get_Y(x, y, z, t, xp, yp, zp, tp, a, b, t_e):
    func = get_f(x, y, z, t, xp, yp, zp, tp) * get_g(z, t, zp, tp) * get_h(xp, yp, zp, tp,  a, b, t_e)
    if z <= d:
        return 1 / (Cv_1 * rho_1) * func
    else:
        return 1 / (Cv_2 * rho_2) * func


# %% MC integration
nmc = 10000

a = b = 0.2e-6  # m
t_exp = 1e-9
# t_exp = 1.64e-9
tt = np.linspace(0.001, 5, 20) * t_exp

# a = b = 0.2e-6  # m
# t_exp = 2e-9  # s
# tt = np.linspace(0.01, 4.01, 20) * 1e-9
results = np.zeros(len(tt))

T_total = np.zeros(len(tt))
err_total = np.zeros(len(tt))
progress_bar = tqdm(total=len(tt), position=0)

for i, now_t in enumerate(tt):

    x_f, y_f, z_f, t_f = 0, 0, 0, now_t
    domainsize = a * b * d * t_f

    def integrand(xx):
        xp = xx[0]
        yp = xx[1]
        zp = xx[2]
        tp = xx[3]
        return get_Y(x_f, y_f, z_f, t_f, xp, yp, zp, tp, a, b, t_exp)

    def sampler():
        while True:
            xp = random.uniform(-a / 2, a / 2)
            yp = random.uniform(-b / 2, b / 2)
            zp = random.uniform(0, d)
            tp = random.uniform(0, t_f)
            yield (xp, yp, zp, tp)

    integral, err = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)
    T_total[i], err_total[i] = integral, err

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.semilogy(tt * 1e+6, T_total, 'o-', label='MC integral')
plt.semilogy(tt * 1e+6, T_total + err_total, 'v-', label='upper limit')
plt.semilogy(tt * 1e+6, T_total - err_total, '^-', label='lower limit')

plt.title(str(Q_uC_cm2) + r' $\mu$C/cm$^2$,' + ' t$_{exp} = $' + str(int(t_exp * 1e+9)) + r' ns, a = b = 200 nm')
plt.xlabel(r't, $\mu$s')
plt.ylabel('T, °C')

plt.xlim(0, tt[-1] * 1e+6)
plt.ylim(0, 10)
plt.legend()
plt.grid()
plt.show()
# plt.savefig('0.02uC_1ns.jpg')
