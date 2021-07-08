import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import nquad
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


def I(x, y, t):
    if np.abs(x) > a/2 or np.abs(y) > b/2 or t > theta:
        return 0
    else:
        return Q / theta


def Y(x, y, z, t, xp, yp, tp):

    mult = f * V / (Cv * rho * Rg) / (4 * alpha * (t - tp))
    exp_1 = np.exp(-(x - xp)**2 / (4 * alpha * (t - tp)))
    exp_2 = np.exp(-(y - yp)**2 / (4 * alpha * (t - tp)))

    body = F(4 * alpha * (t - tp) / Rg**2, z / Rg)

    return mult * I(xp, yp, tp) * body * exp_1 * exp_2


# %% mcint
nmc = 10000

t_coefs = np.linspace(0, 4, 40)
results = np.zeros(len(t_coefs))

progress_bar = tqdm(total=len(t_coefs), position=0)

for i, coef in enumerate(t_coefs):

    x, y, z, t = 0, 0, 0, theta * coef

    domainsize = a * b * t

    def integrand(xx):
        xp = xx[0]
        yp = xx[1]
        tp = xx[2]

        return Y(x, y, z, t, xp, yp, tp)

    def sampler():
        while True:
            xp = random.uniform(-a / 2, a / 2)
            yp = random.uniform(-b / 2, b / 2)
            tp = random.uniform(0, t)
            yield (xp, yp, tp)

    results[i] = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)[0]

    progress_bar.update()

# %%
plt.figure(dpi=300)
plt.plot(t_coefs * theta * 1e+6, results, '.-')

plt.title(r'j=100 A/cm$^2$, D=100 $\mu$C/cm$^2$, 0.5$\times$0.5 $\mu$m$^2$')
plt.xlabel(r't, $\mu$s')
plt.ylabel('delta T, °C')

plt.xlim(0, 4)
plt.ylim(0, 3)

plt.grid()
plt.show()
# plt.savefig('delta_T_2.jpg')
