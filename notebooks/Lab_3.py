import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# %%
def func(x, k, b):
    return k*x + b


U0 = np.array([6, 7, 8])
d1 = np.array([23.8, 22.7, 19.3]) * 1e-3
d2 = np.array([42.6, 36.7, 35.2]) * 1e-3

r1_2_inv = 1 / (d1/2)**2
r2_2_inv = 1 / (d2/2)**2

popt_1, _ = curve_fit(func, U0, r1_2_inv)
popt_2, _ = curve_fit(func, U0, r2_2_inv)

xx = np.linspace(0, 10, 100)
yy_1 = func(xx, *popt_1)
yy_2 = func(xx, *popt_2)


plt.figure(figsize=[4*1.2, 3*1.2], dpi=300)

plt.plot(U0, r1_2_inv, '.', label='1 max')
plt.plot(U0, r2_2_inv, '.', label='2 max')

plt.plot(xx, yy_1)
plt.plot(xx, yy_2)

plt.legend()
plt.xlabel('U$_0$, В')
plt.ylabel(r'$\frac{1}{r^2}$, м$^{-2}$')
plt.xlim(0, 10)
plt.ylim(0, 12e+3)
plt.grid()
plt.show()

# %%
D = 127e-3
h = 6.6262e-34
e = 1.6022e-19
m = 9.1095e-31

gamma_1 = 1838.45
gamma_2 = 512.08

d_1 = D * h * np.sqrt(gamma_1 / (2 * m * e))
d_2 = D * h * np.sqrt(gamma_2 / (2 * m * e))

# %%
U0 = np.array([6, 7, 8]) * 1e+3
d1 = np.array([23.8, 22.7, 19.3]) * 1e-3
d2 = np.array([42.6, 36.7, 35.2]) * 1e-3

r1_2_inv = 1 / (d1/2)**2
r2_2_inv = 1 / (d2/2)**2

popt_1, _ = curve_fit(func, U0, r1_2_inv)
popt_2, _ = curve_fit(func, U0, r2_2_inv)

xx = np.linspace(0, 1e+4, 1000)
yy_1 = func(xx, *popt_1)
yy_2 = func(xx, *popt_2)


plt.figure(figsize=[4*1.2, 3*1.2], dpi=300)

plt.plot(U0, r1_2_inv, '.', label='1 max')
plt.plot(U0, r2_2_inv, '.', label='2 max')

plt.plot(xx, yy_1)
plt.plot(xx, yy_2)

plt.legend()
plt.xlabel('U$_0$, В')
plt.ylabel(r'$\frac{1}{r^2}$, м$^{-2}$')
plt.xlim(0, 1e+4)
plt.ylim(0, 12e+3)
plt.grid()
plt.show()

# %%
D = 127e-3
h = 6.6262e-34
e = 1.6022e-19
m = 9.1095e-31

gamma_1 = 1.838
gamma_2 = 0.512

d_1 = D * h * np.sqrt(gamma_1 / (2 * m * e))
d_2 = D * h * np.sqrt(gamma_2 / (2 * m * e))













