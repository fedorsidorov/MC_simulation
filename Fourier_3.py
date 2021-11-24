import numpy as np
import matplotlib.pyplot as plt


# %%
def get_an(n):
    an = 2/3 * (
        3 / (np.pi * n) * np.sin(np.pi * n / 2) + 3 / (np.pi * n) * np.sin(np.pi * n / 3) -
        np.sin(np.pi * n / 3) * (3 / (np.pi * n) - 54 / (np.pi**3 * n**3)) - 18 / (np.pi**3 * n**3) *
        np.cos(np.pi * n / 3) + 3 / (np.pi * n) * (np.sin(np.pi * n / 2) - np.sin(np.pi * n / 3))
    )
    return an


def get_bn(n):
    bn = 2/3 * (
        -3 / (np.pi * n) * (1 - np.cos(np.pi * n / 2)) +
        3 / (np.pi**3 * n**3) * (-18 * np.cos(np.pi * n / 3) - 6 * np.pi * n * np.sin(np.pi * n / 3) +
                                 np.pi**2 * n**2 + 18) +
        3 / (np.pi * n) * (np.cos(np.pi * n / 3) - np.cos(np.pi * n / 2))
    )
    return bn


# %%
xx_1 = np.linspace(-3/2, 0, 50)
xx_2 = np.linspace(0, 1, 50)
xx_3 = np.linspace(1, 3/2, 50)

yy_1 = np.ones(len(xx_1))
yy_2 = 1 - xx_2**2
yy_3 = np.ones(len(xx_3))

xx_f = np.linspace(-3/2, 3/2, 1000)
yy_f = np.ones(len(xx_f)) * 8/9

for nn in range(1, 101):
    yy_f += get_an(nn) * np.cos(np.pi * nn * xx_f / 3) + get_bn(nn) * np.sin(np.pi * nn * xx_f / 3)

xx = np.concatenate((xx_1, xx_2, xx_3))
yy = np.concatenate((yy_1, yy_2, yy_3))

plt.figure(dpi=300)

plt.plot(xx, yy, label='f(x)')
plt.plot(xx_f, yy_f, label='Fourier series, S$_{100}$(x)')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.ylim(-2, 2)
plt.show()
