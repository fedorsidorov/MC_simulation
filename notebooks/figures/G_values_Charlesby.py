import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# %%
def linear_func(xx, k, b):
    return k * xx + b


TT_C = np.array((-78, 0, 20, 100))
TT_inv = 1 / (TT_C + 273)
TT_C_test = np.linspace(-250, 500, 1000)
TT_test_inv = 1 / (TT_C_test + 273)
GG = np.array((0.74, 1.4, 1.5, 2.3))
GG_log = np.log(GG)
popt, pcov = curve_fit(linear_func, TT_inv, GG_log)


# %%
with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.plot(TT_inv * 1e+3, GG_log, 'o', label='experiment')
    ax.plot(TT_test_inv * 1e+3, linear_func(TT_test_inv, *popt), 'r', label='linear fit')

    ax.legend(fontsize=7)
    ax.set(xlabel=r'T$^{-1}$, K$^{-1}$')
    ax.set(ylabel=r'ln(G)')
    plt.xlim(2, 6)
    plt.ylim(-1, 2)

    plt.show()
    # fig.savefig('figures/lnG_lineat_fit.jpg', dpi=600)


# %%
with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.plot(TT_C, np.exp(GG_log), 'o', label='experiment')
    ax.plot(TT_C_test, np.exp(linear_func(TT_test_inv, *popt)), 'r', label='linear fit')

    ax.legend(fontsize=7)
    ax.set(xlabel=r'T, C')
    ax.set(ylabel=r'G')
    plt.xlim(-100, 150)
    plt.ylim(0, 3)

    plt.show()
    # fig.savefig('figures/G_lineat_fit.jpg', dpi=600)





