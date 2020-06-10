import numpy as np
import matplotlib.pyplot as plt
from functions import MC_functions as mf
from scipy.integrate import quad
from tqdm import tqdm

# %%
profile = np.loadtxt('data/reflow/initial.txt')

prof_x = profile[:, 0] * 1e-6
prof_y = profile[:, 1] * 1e-9


def func(x):
    return mf.lin_lin_interp(prof_x, prof_y)(x)


def An(num, period):
    def get_Y(x):
        return func(x) * np.cos(2 * np.pi * num * x / period)
    return 2 / period * quad(get_Y, -period / 2, period / 2)[0]


def get_tau_n(num):
    if num == 0:
        return np.inf
    part_1 = (num * 2 * np.pi / l_period)**2 * A / (6 * np.pi * h0 * eta)
    part_2 = (num * 2 * np.pi / l_period)**4 * gamma * h0**3 / (3 * eta)
    return 1 / (part_1 + part_2)


def get_tau_n_easy(num):
    if num == 0:
        return np.inf
    return 3 * eta / (gamma * h0**3) * (l_period / (2 * np.pi * num))**4


def get_h(x_array, t):
    result = np.zeros(len(x_array))
    result += An_array[0]
    for num in range(1, N):
        result += An_array[num] * np.exp(-t / get_tau_n_easy(num)) * np.cos(2 * np.pi * num * x_array / l_period)
        # result += An_array[num] * np.exp(-t / get_tau_n(num)) * np.cos(2 * np.pi * num * x_array / l_period)

    return result


# %%
N = 1000
T = 10e-6

xx = np.linspace(-5, 5, 1000) * 1e-6

An_array = np.zeros(N)
An_array[0] = An(0, T) / 2

for n in range(1, N):
    An_array[n] = An(n, T)

# %%
l_period = 10e-6
A = 1e-19  # J
gamma = 34e-3  # N / m
# gamma = 34  # N / m
h0 = 307e-9  # m
# eta = 5.51e+6  # Pa * s
# eta = 5.51e+8  # Pa * s
# eta = 5.64e+8  # Pa * s
eta = 5.48e+8  # Pa * s

# %%
profile = np.loadtxt('data/reflow/2h.txt')

plt.figure(dpi=300)
plt.plot(xx, func(xx))
plt.plot(xx, get_h(xx, 3200*2))

plt.plot(profile[:, 0]*1e-6, profile[:, 1]*1e-9, 'o')

plt.show()
