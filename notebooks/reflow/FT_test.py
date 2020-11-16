import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def f_gate(x, scope=1):
    if np.abs(x) < scope / 2:
        return 1
    else:
        return 0


def f_gate_arr(x_arr):
    result = np.zeros(len(x_arr))
    for i in range(len(x_arr)):
        result[i] = f_gate(x_arr[i])
    return result


def An(func, num, period):
    def get_Y(x):
        return func(x) * np.cos(2 * np.pi * num * x / period)
    return 2 / period * quad(get_Y, -period / 2, period / 2)[0]


def Cn(func, num, period):
    def get_Y_real(x):
        return np.real(func(x) * np.exp(-1j * 2 * np.pi * num * x / period))

    def get_Y_imag(x):
        return np.imag(func(x) * np.exp(-1j * 2 * np.pi * num * x / period))

    real_part = 1 / period * quad(get_Y_real, -period / 2, period / 2)[0]
    imag_part = 1 / period * quad(get_Y_imag, -period / 2, period / 2)[0]
    return real_part + 1j * imag_part


# %%
N = 10
T = 2

A_array = np.zeros(N)
C_array = np.zeros(N, dtype=complex)

xx = np.linspace(-T / 2, T / 2, 1000)

xx_FT_even = np.zeros(len(xx))
xx_FT_even += An(f_gate, 0, T) / 2

for n in range(1, N):
    xx_FT_even += An(f_gate, n, T) * np.cos(np.pi * n * xx / (T / 2))


# %%
xx_FT = np.zeros(len(xx)) + np.zeros(len(xx)) * 1j

for n in range(-N, N + 1):
    xx_FT += Cn(f_gate, n, T) * np.exp(-1j * 2 * np.pi * n * xx / T)


# %%
plt.figure(dpi=300)

plt.plot(xx, f_gate_arr(xx))
plt.plot(xx, xx_FT_even)
plt.plot(xx, np.real(xx_FT), '--')

plt.ylim(-1, 2)

plt.show()
