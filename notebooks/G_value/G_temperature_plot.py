import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_func(xx, k, b):
    return k * xx + b


fig, ax = plt.subplots(dpi=300)
fig.set_size_inches(5, 4)

data = np.loadtxt('data/G_values/gamma.txt')
TT_inv = data[:, 0] / 1e+3
GG = data[:, 1]
GG_log = np.log(GG)
popt, _ = curve_fit(linear_func, TT_inv, GG_log)

plt.semilogy(TT_inv * 1e+3, GG, '*', label='gamma irradiation')
TT_test_inv = 1 / (np.linspace(-50, 10000, 1000) + 273)
plt.semilogy(TT_test_inv * 1e+3, np.exp(linear_func(TT_test_inv, *popt)), '--', label='ln($G_S$)(1/T) linear fit')


TT_C = np.array((-78, 0, 20, 100))
TT_inv = 1 / (TT_C + 273)
GG = np.array((0.74, 1.4, 1.5, 2.3))
GG_log = np.log(GG)
popt, _ = curve_fit(linear_func, TT_inv, GG_log)

plt.semilogy(TT_inv * 1e+3, GG, 'o', label='e-beam irradiation')
TT_test_inv = 1 / (np.linspace(-200, 10000, 1000) + 273)
plt.semilogy(TT_test_inv * 1e+3, np.exp(linear_func(TT_test_inv, *popt)), label='ln($G_S$)(1/T) linear fit')

plt.xticks(np.array((2, 3, 4, 5, 6)), ('2', '3', '4', '5', '6'))
plt.yticks(np.array((0.6, 1, 2, 3, 4)), ('0.6', '1', '2', '3', '4'))

plt.xlabel('100/T, K$^{-1}$')
plt.ylabel('$G_S$')

plt.legend()
plt.grid()
plt.xlim(2, 6)
plt.ylim(0.6, 4)
plt.show()

# %%
# plt.savefig('G_fit.tiff')
