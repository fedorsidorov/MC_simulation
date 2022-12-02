import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# %%
def func(xx, alpha):
    return np.exp(-alpha * xx)


# dd = np.linspace(0, 35, 70)
dd = np.linspace(0, 35, 14)

l_norm = [1, 0.5, 0]

doses_118_1 = [0, 3.8, 30]
doses_98_6 = [0, 22, 180]
doses_125_6 = [0, 5.2, 46]
doses_125_0p15 = [0, 0.85, 8]

doses_125_150nm = [0, 1.3, 10.2]
doses_125_1um = [0, 20, 170]
doses_150_1um = [0, 3.6, 30]
doses_170_2um = [0, 4.0, 33.6]

plt.figure(dpi=300)

plt.plot(doses_118_1, l_norm, 'o')
plt.plot(dd, func(dd, curve_fit(func, doses_118_1, l_norm)[0]), label='118 °C, 1 nA')

plt.plot(doses_98_6, l_norm, 'o')
plt.plot(dd, func(dd, 0.0315), label='98 °C, 6 nA')
# plt.plot(dd, func(dd, curve_fit(func, doses_98_6, l_norm)[0]), label='98 °C, 6 nA')

plt.plot(doses_125_6, l_norm, 'o')
# plt.plot(dd, func(dd, 0.133), label='125 °C, 6 nA')
plt.plot(dd, func(dd, curve_fit(func, doses_125_6, l_norm)[0]), label='125 °C, 6 nA')

plt.plot(doses_125_0p15, l_norm, 'o')
plt.plot(dd, func(dd, curve_fit(func, doses_125_0p15, l_norm)[0]), label='125 °C, 0.15 nA')

plt.plot(doses_125_150nm, l_norm, 'o')
plt.plot(dd, func(dd, curve_fit(func, doses_125_150nm, l_norm)[0]), label='125 °C, 1 nA, 150 nm')

plt.plot(doses_125_1um, l_norm, 'o')
plt.plot(dd, func(dd, curve_fit(func, doses_125_1um, l_norm)[0]), label='125 °C, 1 nA, 1 um')

plt.plot(doses_150_1um, l_norm, 'o')
plt.plot(dd, func(dd, curve_fit(func, doses_150_1um, l_norm)[0]), label='150 °C, 1 nA, 1 um')

plt.plot(doses_170_2um, l_norm, 'o')
plt.plot(dd, func(dd, curve_fit(func, doses_170_2um, l_norm)[0]), label='170 °C, 1 nA, 2 um')

plt.xlabel(r'15now21, $\mu$C/cm$^2$')
plt.ylabel(r'L/L$_0$')
# plt.xlim(0, 35)
plt.ylim(0, 1.2)
plt.grid()
plt.legend()
plt.show()
