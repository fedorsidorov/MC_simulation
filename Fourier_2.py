import numpy as np
import matplotlib.pyplot as plt


# %%
def get_bn(n):
    bn = 2/3 * (
        3 / (np.pi**3 * n**3) * (-18 * np.cos(np.pi * n / 3) - 6 * np.pi * n * np.sin(np.pi * n / 3) +
                                 np.pi**2 * n**2 + 18) +
        3 / (np.pi * n) * (np.cos(np.pi * n / 3) - (-1)**n)
    )
    return bn


# %%
xx_1 = np.linspace(-3, -1, 50)
xx_2 = np.linspace(-1, 0, 50)
xx_3 = np.linspace(0, 1, 50)
xx_4 = np.linspace(1, 3, 50)

yy_1 = -np.ones(len(xx_1))
yy_2 = -(1 - xx_2**2)
yy_3 = 1 - xx_3**2
yy_4 = np.ones(len(xx_4))

xx_f = np.linspace(-3, 3, 1000)
yy_f = np.zeros(len(xx_f))

for nn in range(1, 101):
    yy_f += get_bn(nn) * np.sin(np.pi * nn * xx_f / 3)

xx = np.concatenate((xx_1, xx_2, xx_3, xx_4))
yy = np.concatenate((yy_1, yy_2, yy_3, yy_4))

plt.figure(dpi=300)

plt.plot(xx, yy, label='f(x)')
plt.plot(xx_f, yy_f, label='Fourier series, S$_{100}$(x)')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.ylim(-2, 2)
plt.show()
