import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 12})


# %%
def linear_func(xx, C, gamma):
    return C * xx**gamma


etas_SI = np.array((1e+2, 3.1e+2, 1e+3, 3.1e+3, 1e+4, 3.1e+4, 1e+5, 3.1e+5, 1e+6))
alphas = np.array((3.62, 1.11e+1, 3.58e+1, 1.1e+2, 3.61e+2, 1.1e+3, 3.48e+3, 1.11e+4, 3.53e+4))

popt, pcov = curve_fit(linear_func, etas_SI, alphas)

xx = np.linspace(etas_SI[0], etas_SI[-1], 1000)
yy = linear_func(xx, *popt)

# plt.figure(dpi=600, figsize=[6.4, 4.8])
plt.figure(dpi=600, figsize=[6.4 / 1.9, 4.8 / 1.9])
plt.loglog(etas_SI, alphas, '.', label=r'simulation русский язык', )
plt.loglog(xx, yy, 'r', label=r'$\alpha$ = $C\eta^\beta$ fit')

plt.loglog([1], [1], 'w', label=r'C=' + str(format(popt[0], '.2e')))
plt.loglog([1], [1], 'w', label=r'$\beta$=' + str(format(popt[1], '.3')))

plt.xlabel(r'$\eta$, Pa$\cdot$s русский язык')
plt.ylabel(r'$\alpha$ = time/scale')
plt.legend(loc='upper left', fontsize=10)

plt.xlim(3e+1, 3e+6)
plt.ylim(1e+0, 1e+6)
plt.grid()

# plt.show()
# plt.savefig('figures/С_gamma_600_1.jpg', dpi=600, bbox_inches='tight')
plt.savefig('figures/С_gamma_600_1p9.jpg', dpi=600, bbox_inches='tight')

# %
# with plt.style.context(['science', 'grid', 'russian-font']):
#     fig, ax = plt.subplots(dpi=600)
#
#     ax.loglog(etas_SI, alphas, '.', label=r'simulation русский язык')
#     ax.loglog(xx, yy, 'r', label=r'$\alpha$ = $C\eta^\beta$ fit')
#
#     ax.loglog([1], [1], 'w', label=r'C=' + str(format(popt[0], '.2e')))
#     ax.loglog([1], [1], 'w', label=r'$\beta$=' + str(format(popt[1], '.3')))
#
#     # ax.set(title=r'$\eta$ = ' + str(format(now_eta, '.1e')) + ' Pa$\cdot$s')
#     ax.set(xlabel=r'$\eta$, Pa$\cdot$s русский язык')
#     ax.set(ylabel=r'$\alpha$ = time/scale')
#     ax.legend(fontsize=7, loc='upper left')
#
#     plt.xlim(3e+1, 3e+6)
#     plt.ylim(1e+0, 1e+6)
#
#     # plt.savefig('figures/С_gamma.jpg', dpi=600)
#     plt.savefig('figures/С_gamma.pdf', dpi=600)
#     plt.show()
