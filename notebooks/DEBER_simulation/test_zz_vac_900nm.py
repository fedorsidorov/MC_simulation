import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np

# %%
file_0p05 = np.loadtxt('notebooks/DEBER_simulation/exp_curves/exp_900nm_5um_0.05uC_cm2.txt')
file_0p2 = np.loadtxt('notebooks/DEBER_simulation/exp_curves/exp_900nm_5um_0.2uC_cm2.txt')
file_0p87 = np.loadtxt('notebooks/DEBER_simulation/exp_curves/exp_900nm_5um_0.87uC_cm2.txt')

plt.figure(dpi=300)
# plt.plot(file_0p05[:, 0], file_0p05[:, 1])
plt.plot(file_0p2[:, 0], file_0p2[:, 1])
# plt.ylim(0, 0.2)
plt.grid()
plt.show()

dz_1 = (file_0p05[:, 1].max() - file_0p05[:, 1].min()) * 1000
dz_2 = (file_0p2[:, 1].max() - file_0p2[:, 1].min()) * 1000
dz_3 = (file_0p87[:, 1].max() - file_0p87[:, 1].min()) * 1000

plt.figure(dpi=300)
plt.plot([0, 0.05, 0.2, 0.87], [0, dz_1, dz_2, dz_3], '*-')

plt.ylim(0, 1000)
plt.xlim(0, 1)

plt.grid()

plt.show()

