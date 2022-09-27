import importlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# %%
def linear_func(xx, C, gamma):
    return C / xx**gamma


etas_SI = np.array([1e+2, 3.1e+2, 1e+3, 3.1e+3, 1e+4, 3.1e+4, 1e+5, 3.1e+5, 1e+6])
alphas = np.array([2.75e-1, 9.01e-2, 2.79e-2, 9.08e-3, 2.77e-3, 9.10e-4, 2.87e-4, 9.01e-5, 2.83e-5])

popt, pcov = curve_fit(linear_func, etas_SI, alphas)

xx = np.linspace(1e+1, 1e+7, 1000)
yy = linear_func(xx, *popt)


with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.loglog(etas_SI, alphas, '.', label=r'simulation')
    ax.loglog(xx, yy, 'r', label=r'mobility = $C/\eta^\beta$ fit')

    ax.loglog([1], [1], 'w', label=r'C=' + str(format(popt[0], '.3e')))
    ax.loglog([1], [1], 'w', label=r'$\beta$=' + str(format(popt[1], '.3')))

    # ax.set(title=r'$\eta$ = ' + str(format(now_eta, '.1e')) + ' Pa$\cdot$s')
    ax.set(xlabel=r'$\eta$, Pa$\cdot$s')
    ax.set(ylabel=r'mobility = scale/time')
    ax.legend(fontsize=7, loc='upper right')

    plt.xlim(1e+1, 1e+7)
    plt.ylim(1e-5, 1e+1)

    plt.savefig('figures/С_gamma.jpg', dpi=600)
    # plt.show()
