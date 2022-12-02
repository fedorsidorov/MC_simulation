import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_func(xx, k, b):
    return k * xx + b


# %%
weights = np.array((0.5, 0.4, 0.3, 0.2, 0.1))
temps_C = np.array((50, 60, 70, 80))

D_matrix = np.zeros((len(temps_C), len(weights)))

D_matrix[0, :] = 2.4, 16.5, 34.5, 37.9, 39
D_matrix[1, :] = 6.2, 19.9, 55.0, 50.1, 55.5
D_matrix[2, :] = 11.6, 35.7, 79.9, 76.9, 66.7
D_matrix[3, :] = 16.0, 40.0, 103.1, 99.0, 105.7

D_matrix *= 1e-7

# %%
plt.figure(dpi=300)

for i in range(len(temps_C)):
    plt.plot(weights, np.log10(D_matrix[i, :]), label=str(int(temps_C[i])))

plt.xlabel('PMMA weight')
plt.ylabel('log(D)')
plt.legend()
plt.grid()
plt.show()

# %%
plt.figure(dpi=300)

for j in range(len(weights)):
    plt.plot(1000 / (temps_C + 273), np.log10(D_matrix[:, j]), label=str(int(weights[j] * 100)) + '%')

plt.xlabel('1000/T, K$^{-1}$')
plt.ylabel('log(D)')
plt.legend()
plt.grid()
plt.show()

# %%
popt, _ = curve_fit(linear_func, 1000 / (temps_C + 273), np.log10(D_matrix[:, -1]))

print(linear_func(1000 / (125 + 273), *popt))


# %%
def get_log_D(wp, dT, C_array):
    C1, C2, C3, C4 = C_array
    log_D = wp * dT * C4 + (C1 - wp * C2) + dT * C3
    return log_D


# %%
coefs = np.zeros((4, 4))

coefs[0, :] = -4.428, 1.842, 0, 8.12e-3
coefs[1, :] = 26.0, 37.0, 0.0797, 0
coefs[2, :] = 159.0, 170.0, 0.3664, 0
coefs[3, :] = -13.7, 0.5, 0, 0

# w_pol = np.linspace(0, 1, 100)
# delta_T = np.linspace(0, 50, 100)
delta_T = 30
# delta_T = 0
w_pol = 1
# w_pol = 0.95

plt.figure(dpi=300)

# plt.plot(w_pol, get_log_D(w_pol, delta_T, coefs[0, :]), label='1')
# plt.plot(w_pol, get_log_D(w_pol, delta_T, coefs[1, :]), label='2')
# plt.plot(w_pol, get_log_D(w_pol, delta_T, coefs[2, :]), label='3')
# plt.plot(w_pol, get_log_D(w_pol, delta_T, coefs[3, :]), label='4')

# plt.plot(delta_T, get_log_D(w_pol, delta_T, coefs[0, :]), label='1')
plt.plot(delta_T, get_log_D(w_pol, delta_T, coefs[1, :]), label='2')  # region of interest
# plt.plot(delta_T, get_log_D(w_pol, delta_T, coefs[2, :]), label='3')
# plt.plot(delta_T, get_log_D(w_pol, delta_T, coefs[3, :]), label='4')

# plt.xlim(0.75, 1)
# plt.ylim(-15, 0)

plt.grid()
plt.legend()

plt.show()


