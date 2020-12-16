import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functions import boyd_functions as bf

bf = importlib.reload(bf)


def func(xx, A, k):
    return A*np.exp(k/xx)


temps = np.array([140, 160, 180]) + 273
TT = np.linspace(90, 190, 50) + 273

k_d = [3e+3, 8.2e+3, 2.1e+4]
k_t = [3.8, 5.9, 6.6]
k_f = [3.2, 4.9, 5.5]

zip_lens_term = [7.9e+2, 1.4e+3, 3.2e+3]
zip_lens_trans = [9.4e+2, 1.7e+3, 3.8e+3]

popt_k_d = curve_fit(func, temps, k_d)[0]
popt_k_t = curve_fit(func, temps, k_t)[0]
popt_k_f = curve_fit(func, temps, k_f)[0]

plt.figure(dpi=300)

# plt.plot(temps, k_d, 'o')
# plt.plot(temps, k_t, 'o')
# plt.plot(temps, k_f, 'o')

# plt.plot(TT, func(TT, *popt_k_d))
# plt.plot(TT, func(TT, *popt_k_t))
# plt.plot(TT, func(TT, *popt_k_f))

plt.plot(temps - 273, zip_lens_term, 'o')
plt.plot(TT - 273, func(TT, *popt_k_d) / func(TT, *popt_k_t), label=r'$k_d / k_t$')

plt.plot(temps - 273, zip_lens_trans, 'o')
plt.plot(TT - 273, func(TT, *popt_k_d) / func(TT, *popt_k_f))

zips_term, zips_trans = bf.get_zip_len_term_trans(temps - 273)

plt.plot(temps - 273, zips_term, '*')
plt.plot(temps - 273, zips_trans, '*')

plt.xlim(100, 200)
plt.ylim(0, 4000)

plt.legend()
plt.grid()
plt.show()
