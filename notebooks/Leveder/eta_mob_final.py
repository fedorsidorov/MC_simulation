import importlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

fontsize = 12
font = {'size': fontsize}
matplotlib.rc('font', **font)


# %%
def linear_func(xx, C, gamma):
    return C / xx**gamma


etas_SI = np.array([1e+2, 3.1e+2, 1e+3, 3.1e+3, 1e+4, 3.1e+4, 1e+5, 3.1e+5, 1e+6])
alphas = np.array([2.75e-1, 9.01e-2, 2.79e-2, 9.08e-3, 2.77e-3, 9.10e-4, 2.87e-4, 9.01e-5, 2.83e-5])

popt, pcov = curve_fit(linear_func, etas_SI, alphas)

xx = np.linspace(1e+1, 1e+7, 1000)
yy = linear_func(xx, *popt)

plt.figure(dpi=600, figsize=[4, 3])

plt.loglog(etas_SI, alphas, '.', label=r'моделирование')
plt.loglog(xx, yy, 'r', label=r'$\mu = C/\eta^\beta$')

# plt.loglog([1], [1], 'w', label=r'$C$ = ' + str(format(popt[0], '.3e')))
plt.loglog([1], [1], 'w', label=r'$C = 26.1416$')
# plt.loglog([1], [1], 'w', label=r'$\beta$=' + str(format(popt[1], '.3')))
plt.loglog([1], [1], 'w', label=r'$\beta = 0.9889$')

plt.xlabel(r'$\eta$, Па$\cdot$с')
plt.ylabel(r'$\mu = s/t$')
plt.legend(fontsize=10, loc='upper right')

plt.grid()
plt.xlim(1e+1, 1e+7)
plt.ylim(1e-5, 1e+1)

plt.savefig('С_gamma_' + str(fontsize) + '.jpg', dpi=600, bbox_inches='tight')
# plt.show()



