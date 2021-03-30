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
k = 0.2  # W / m / K
Cv = 1500  # J / kg / K
alpha = k / (Cv * rho)

V = 40e+3
j = 100 * 1e+4  # A / m^2
Q = 100e-6 * 1e+4  # C / m^2
theta = 1e-6
Rg = 20e-6  # m
lambda_z_Rg = 1

# a = b = 0.5e-6  # m
a = b = 1e-6  # m
d = 1e-6  # m

n_terms = 50


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

    for m in range(1, n_terms + 1):
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
    if np.abs(xp) > a/2 or np.abs(yp) > b/2 or zp > d or tp > theta:
        return 0
    else:
        return V * Q * lambda_z_Rg / (Rg * theta)


# %%
nmc = 10000

t_coefs = np.linspace(0.01, 4.01, 40)
results = np.zeros(len(t_coefs))

# %% mcint
progress_bar = tqdm(total=len(t_coefs), position=0)

for i, coef in enumerate(t_coefs):

    x_f, y_f, z_f, t_f = 0, 0, 0, theta * coef
    domainsize = a * b * d * t_f

    def integrand(xx):
        xp = xx[0]
        yp = xx[1]
        zp = xx[2]
        tp = xx[3]

        return G(x_f, y_f, z_f, t_f, xp, yp, zp, tp) * h(xp, yp, zp, tp)

    def sampler():
        while True:
            xp = random.uniform(-a / 2, a / 2)
            yp = random.uniform(-b / 2, b / 2)
            zp = random.uniform(0, d)
            tp = random.uniform(0, t_f)
            yield (xp, yp, zp, tp)

    results[i] = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)[0]

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.plot(t_coefs, results / Cv / rho)
plt.show()





