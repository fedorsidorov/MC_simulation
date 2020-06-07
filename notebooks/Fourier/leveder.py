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


def Cn(num, period):
    def get_Y_real(x):
        return np.real(func(x) * np.exp(-1j * 2 * np.pi * num * x / period))

    def get_Y_imag(x):
        return np.imag(func(x) * np.exp(-1j * 2 * np.pi * num * x / period))

    real_part = 1 / period * quad(get_Y_real, -period / 2, period / 2)[0]
    imag_part = 1 / period * quad(get_Y_imag, -period / 2, period / 2)[0]
    return real_part + 1j * imag_part


# %%
N = 1000
T = 10e-6

xx = np.linspace(-5, 5, 1000) * 1e-6

An_array = np.zeros(N)

xx_FT_even = np.zeros(len(xx))
xx_FT_even += An(0, T) / 2
An_array[0] = An(0, T) / 2

for n in range(1, N):
    An_array[n] = An(n, T)
    # xx_FT_even += An(n, T) * np.cos(np.pi * n * xx / (T / 2))

# %%
FT = np.zeros(len(xx)) + np.zeros(len(xx)) * 1j
Cn_array = np.zeros(2 * N + 1, dtype=complex)

progress_bar = tqdm(total=2*N+1, position=0)

for n in range(-N, N + 1):
    # FT += Cn(n, T) * np.exp(1j * 2 * np.pi * n * xx / T)
    Cn_array[n] = Cn(n, T)
    progress_bar.update(1)

# %%
# l_period = 10e-6
l_period = 500e-9
A = 1e-19  # J
gamma = 34e-3  # N / m
# h0 = 307e-9  # m
h0 = 4e-9  # m
eta = 5.51e+6  # Pa * s


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
    result +=
    for num in range(1, N):
        result += An_array[num] * np.exp(-t / get_tau_n(num)) * np.cos(2 * np.pi * num * x_array / l_period)

    return result


def get_h_complex(x_array, t):
    result = np.zeros(len(x_array))
    for num in range(-N, N + 1):
        result += np.real(Cn_array[num] * np.exp(-t / get_tau_n(num) + 1j * num * 2 * np.pi * x_array / l_period))

    return result


# %%
plt.figure(dpi=300)
plt.plot(xx, func(xx))
# plt.plot(xx, xx_FT_even)
plt.plot(xx, np.real(FT))
plt.plot(xx, get_h(xx, 2))
plt.show()
