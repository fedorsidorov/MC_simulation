#%% Import
import numpy as np
import matplotlib.pyplot as plt
import importlib

import constants as const
const = importlib.reload(const)


#%%
def get_Mn(x_mw, y):
    return np.sum(y * x_mw) / np.sum(y)


def get_Mw(x_mw, y):
    return np.sum(y * np.power(x_mw, 2)) / np.sum(y * x_mw)


def Flory_Schulz(k, n, p):
    return n * np.power(1-p, 2) * k * np.power(p, k-1)


def Gauss(k, n_g, mu, sigma):
    return n_g * np.exp(-(k - mu)**2 / (2 * sigma**2))


def Flory_Schulz_mod(k, n, p, n_g, mu, sigma):
    return n * np.power(1-p, 2) * k * np.power(p, k-1) + Gauss(k, n_g, mu, sigma)


#%%
mma_mass = const.u_MMA
Mn_0 = 27.1e+4  # from e-mail
Mw_0 = 66.9e+4  # from e-mail

x_Mn_Mw = np.ones(100)
y_Mn_Mw = np.linspace(0, 10, len(x_Mn_Mw))

N = 1e+6
P = 99.998963e-2
N_G = 0.1
MU = 1e+5
SIGMA = 1.008e+6

params = N, P, N_G, MU, SIGMA

# x_FS = np.logspace(2, 7, 500)
x_FS = np.linspace(10**2, 10**7, 5000)
y_FS = Flory_Schulz_mod(x_FS, *params)

x_Mn_Mw = np.ones(100)
# y_Mn_Mw = np.linspace(0, 3, 50)

Mn_FS = get_Mn(x_FS, y_FS)
Mw_FS = get_Mw(x_FS, y_FS)

plt.figure(dpi=300)
plt.semilogx(x_FS, y_FS, 'ro', label='Flory-Schulz + Gauss')
plt.semilogx(x_Mn_Mw * Mn_FS, y_Mn_Mw, label='Mn_model')
plt.semilogx(x_Mn_Mw * Mw_FS, y_Mn_Mw, label='Mw_model')
plt.semilogx(x_Mn_Mw * Mn_0, y_Mn_Mw, '--', label='Mn_exp')
plt.semilogx(x_Mn_Mw * Mw_0, y_Mn_Mw, '--', label='Mw_exp')

# plt.semilogx(x_FS, Flory_Schulz(x_FS, N, P), label='Flory-Schulz')
# plt.semilogx(x_FS, Gauss(x_FS, N_G, MU, SIGMA), label='Gauss')

# plt.xlim(0, 1e+6)
plt.legend()
# plt.gca().get_xaxis().get_major_formatter().set_powerlimits((0, 0))
plt.grid()
plt.show()
