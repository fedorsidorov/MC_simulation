import numpy as np
import matplotlib.pyplot as plt

# %%
T = 100
h_bar = 1.05e-34
k = 1.38e-23
c = 3e+8

w_0 = k * T / h_bar

ww = np.array((1/4, 1/2, 1, 2, 3, 4, 6, 8)) * w_0
ff = h_bar * ww**3 / (4 * np.pi**2 * c**2 * (np.exp(h_bar * ww / k / T - 1)))

with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)
    ax.plot(ww, ff, 'o')

    plt.xlim(0, 1.2e+14)
    plt.ylim(0, 3e-13)
    ax.set(xlabel=r'$\omega$, 1/c')
    ax.set(ylabel=r'$f(\omega)$, Дж/м$^2$)')

    plt.show()

