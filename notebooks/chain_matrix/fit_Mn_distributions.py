import importlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# %%
def func(x, a, b, c):
    return 1 - a / (np.exp((np.log(x) - b)/c) + 1)


# %%
# before
Mn = 5.63e+5
Mw = 2.26e+6

# after
# Mn = 2370
# Mw = 8160

Harris_data = np.loadtxt('data/chains/Harris/curves/Harris_before.txt')
xx_fit = np.logspace(2, 8, 1001)  # before

# mat = np.loadtxt('data/Harris/curves/Harris_after.txt')
# x_fit = np.logspace(2, 6, 10000)  # after

# %%
xx = Harris_data[:, 0]
yy = Harris_data[:, 1]

popt, pcov = curve_fit(func, xx, yy)

y_fit = func(xx_fit, *popt)

plt.semilogx(xx_fit, y_fit, 'b-', label='fit')
plt.semilogx(xx, yy, 'ro', label='paper data')
plt.title('Harris final integral molecular weight distribution')
plt.xlabel('molecular weight')
plt.ylabel('Distribution function')

plt.legend()
plt.grid()
plt.show()

#%%
x_diff = xx_fit[:-1]
y_diff = np.diff(y_fit)
y_diff_n = y_diff / np.max(y_diff)

plt.semilogx(x_diff, y_diff_n, label='fit')
plt.title('Harris initial molecular weight distribution')
plt.xlabel('molecular weight')
plt.ylabel('density')

plt.legend()
plt.grid()
plt.show()

#%%
distr = np.array(list(zip(x_diff, y_diff_n)))
plt.semilogx(distr[:, 0], distr[:, 1])
plt.show()

# plt.savefig('Harris_after.png', dpi=300)

#%%
Mn_fit = np.dot(x_diff, y_diff_n) / np.sum(y_diff_n)
Mw_fit = np.dot(np.power(x_diff, 2), y_diff_n) / np.dot(x_diff, y_diff_n)

names = 'Mn', 'Mw'

plt.plot(names, [Mn_fit, Mw_fit], '^-', label='fit')
plt.plot(names, [Mn, Mw], 'o-', label='paper')

plt.title('Harris final Mn and Mw')
plt.ylabel('average M')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Mn_Mw_final.png', dpi=300)
