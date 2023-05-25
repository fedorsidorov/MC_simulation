import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'size': 14}
matplotlib.rc('font', **font)


def linear_func(xx, k, b):
    return k * xx + b


plt.figure(dpi=300, figsize=[4, 3])

data = np.loadtxt('/Users/fedor/PycharmProjects/MC_simulation/data/_outdated/G_values/gamma.txt')
TT_inv = data[:, 0] / 1e+3
GG = data[:, 1]
GG_log = np.log(GG)
popt, _ = curve_fit(linear_func, TT_inv, GG_log)

TT_test_inv = 1 / (np.linspace(-50, 10000, 1000) + 273)
plt.semilogy(TT_test_inv * 1e+3, np.exp(linear_func(TT_test_inv, *popt)), color='C1')
plt.semilogy(TT_inv * 1e+3, GG, 'C0*', label=r'$\gamma$-излучение')
# plt.semilogy(TT_test_inv * 1e+3, np.exp(linear_func(TT_test_inv, *popt)), '--', label=r'функция ln($G_S$)(1/T)')

TT_C = np.array((-78, 0, 20, 100))
TT_inv = 1 / (TT_C + 273)
GG = np.array((0.74, 1.4, 1.5, 2.3))
GG_log = np.log(GG)
popt, _ = curve_fit(linear_func, TT_inv, GG_log)

TT_test_inv = 1 / (np.linspace(-200, 10000, 1000) + 273)
plt.semilogy(TT_test_inv * 1e+3, np.exp(linear_func(TT_test_inv, *popt)), color='C3')
plt.semilogy(TT_inv * 1e+3, GG, 'C2o', label='электронный луч')
# plt.semilogy(TT_test_inv * 1e+3, np.exp(linear_func(TT_test_inv, *popt)), label=r'функция ln($G_S$)(1/T)')

plt.xticks(np.array((2, 3, 4, 5, 6)), ('2', '3', '4', '5', '6'))
plt.yticks(np.array((0.6, 1, 2, 3, 4)), ('0.6', '1', '2', '3', '4'))

# plt.xlabel(r'$\frac{100}{T}$, K$^{-1}$')
plt.xlabel(r'$1000/T$, K$^{-1}$')
plt.ylabel(r'$G_\mathrm{s}$')

# plt.legend(fontsize=12, loc='upper right')
plt.legend(fontsize=12, loc='lower left')
plt.grid()
# plt.xlim(2, 6)
plt.xlim(1, 6)
plt.ylim(0.6, 5)

plt.savefig('Charlesby_G_fit_1_CORR.jpg', dpi=300, bbox_inches='tight')
plt.show()
