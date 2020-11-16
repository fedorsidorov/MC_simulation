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
