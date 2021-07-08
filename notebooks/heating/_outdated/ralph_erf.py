import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import nquad
from scipy.special import erf
from scipy.integrate import quad
import mcint
import random

from functions import MC_functions as mcf
import grid

grid = importlib.reload(grid)
mcf = importlib.reload(mcf)

# %% simple_Si_MC
e = eV = 1.6e-19

rho = 2330  # kg / m^3
k = 150  # W / m / K
Cv = 700  # J / kg / K
alpha = k / (Cv * rho)

V = 40e+3
j = 100 * 1e+4  # A / m^2
Q = 100e-6 * 1e+4  # C / m^2
theta = 1e-6
a = b = 0.5e-6  # m

Rg = 20e-6  # m
lambda_z_Rg = 1
f = 1


# %%
def F(tau, dzeta):

    def F_Y(ksi):
        return lambda_z_Rg * (np.exp(-(dzeta - ksi)**2 / tau) + np.exp(-(dzeta + ksi)**2 / tau))

    return quad(F_Y, 0, 10)[0]


def G(tau, dzeta):
    return (erf((1 - dzeta) / np.sqrt(tau)) + erf(dzeta / np.sqrt(tau))) / 2


def I(x, y, t):
    if np.abs(x) > a/2 or np.abs(y) > b/2 or t > theta:
        return 0
    else:
        return Q / theta


def S(arg):
    if arg < 1:
        return 1
    return 0


def Y(x, y, z, t, tp):

    t0 = f * V * Q / (Cv * rho * Rg)
    t1 = S(tp / theta) / theta
    t2 = F(4 * alpha * (t - tp) / Rg**2, z / Rg)
    t3 = G(4 * alpha * (t - tp) / a**2, x / a)
    t4 = G(4 * alpha * (t - tp) / b**2, y / b)

    return t0 * t1 * t2 * t3 * t4


# %% mcint
domainsize = a * b * theta

nmc = 10000

t_coefs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01,
           1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
results = np.zeros(len(t_coefs))

progress_bar = tqdm(total=len(t_coefs), position=0)

for i, coef in enumerate(t_coefs):

    x_f, y_f, z_f, t_f = 0, 0, 0, theta * coef

    def integrand(xx):
        tp = xx[0]
        pp = xx[0]
        return Y(x_f, y_f, z_f, t_f, tp)

    def sampler():
        while True:
            if t_f > theta:
                return random.uniform(0, theta), 0
            return random.uniform(0, t_f), 0

    results[i] = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)[0]

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.plot(t_coefs, results)
plt.show()

