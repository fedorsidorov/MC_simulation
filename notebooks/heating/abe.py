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

rho = 1200  # kg / m^3
# D = 0.2  # W / m / K
k = 0.2  # W / m / K
Cv = 1500  # J / kg / K
alpha = k / (Cv * rho)
# K_2 = D / (Cv * rho)
# K = np.sqrt(K_2)

V = 40e+3
j = 100 * 1e+4  # A / m^2
Q = 100e-6 * 1e+4  # C / m^2
d = 1e-6  # m
theta = 1e-6
Rg = 20e-6  # m
lambda_z_Rg = 1

lx = ly = 0.5e-6  # m


# %%
def f(x, t, xp, tp):
    mult = 1 / (np.sqrt(4 * alpha * np.pi * (t - tp)))
    exp = np.exp(-(x - xp)**2 / (4 * alpha * (t - tp)))
    return mult * exp


def g(z, t, zp, tp):
    mult = 1 / (np.sqrt(4 * alpha * np.pi * (t - tp)))

    exp_1 = np.exp(-(z - zp)**2 / (4 * alpha * (t - tp)))
    exp_2 = np.exp(-(z + zp)**2 / (4 * alpha * (t - tp)))

    total_sum = 0

    for m in range(1, 11):
        term_1 = np.exp(-(2 * m * d - z - zp)**2 / (4 * alpha * (t - tp)))
        term_2 = np.exp(-(2 * m * d + z - zp)**2 / (4 * alpha * (t - tp)))
        term_3 = np.exp(-(-2 * m * d + z - zp)**2 / (4 * alpha * (t - tp)))
        term_4 = np.exp(-(-2 * m * d - z - zp)**2 / (4 * alpha * (t - tp)))

        total_sum += (-1)**m * (term_1 + term_2 + term_3 + term_4)

    return mult * (exp_1 + exp_2 + total_sum)


def G(x, y, z, t, xp, yp, zp, tp):
    if tp > t:
        return 0
    else:
        return f(x, t, xp, tp) * f(y, t, yp, tp) * g(z, t, zp, tp)


def h(xp, yp, zp, tp):
    if np.abs(xp) > lx/2 or np.abs(yp) > ly/2 or zp > d or tp > theta:
        return 0
    else:
        return V * Q * lambda_z_Rg / (Rg * theta)


# %%
nmc = 100000

t_coefs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
results = np.zeros(len(t_coefs))


# %% nquad
progress_bar = tqdm(total=len(t_coefs), position=0)

for i, coef in enumerate(t_coefs):

    x, y, z, t = 0, 0, 0, theta * coef
    domainsize = lx * ly * d * t

    def integrand(xp, yp, zp, tp):
        return G(x, y, z, t, xp, yp, zp, tp) * h(xp, yp, zp, tp)

    results[i] = nquad(integrand, [
        [-lx, lx],
        [-ly, ly],
        [0, d],
        [0, t]
    ])[0]

    progress_bar.update()


# %% mcint
progress_bar = tqdm(total=len(t_coefs), position=0)

for i, coef in enumerate(t_coefs):

    x, y, z, t = 0, 0, 0, theta * coef

    def integrand(xx):
        xp = xx[0]
        yp = xx[1]
        zp = xx[2]
        tp = xx[3]

        return G(x, y, z, t, xp, yp, zp, tp) * h(xp, yp, zp, tp)

    def sampler():
        while True:
            xp = random.uniform(-lx / 2, lx / 2)
            yp = random.uniform(-ly / 2, ly / 2)
            zp = random.uniform(0, d)
            tp = random.uniform(0, t)
            yield (xp, yp, zp, tp)

    results[i] = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)[0]

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.plot(t_coefs, results / Cv / rho)
plt.show()





