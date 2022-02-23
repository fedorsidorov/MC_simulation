import numpy as np
import matplotlib.pyplot as plt

# %% A
A1y = np.loadtxt('notebooks/DEBER_profiles/Fedor/356y/height/A/1d.csv', delimiter=',', skiprows=5)
B1y = np.loadtxt('notebooks/DEBER_profiles/Fedor/356y/B/1d.csv', delimiter=',', skiprows=5)
C1y = np.loadtxt('notebooks/DEBER_profiles/Fedor/356y/C/1d.csv', delimiter=',', skiprows=5)

A1y = np.loadtxt('notebooks/DEBER_profiles/Fedor/356/C_height_1/C1_1.csv', delimiter=',', skiprows=5)
B1y = np.loadtxt('notebooks/DEBER_profiles/Fedor/356/C_height_1/C1_2.csv', delimiter=',', skiprows=5)
C1y = np.loadtxt('notebooks/DEBER_profiles/Fedor/356/C_height_1/C1_3.csv', delimiter=',', skiprows=5)


plt.figure(dpi=300)

plt.plot(A1y[:, 0], A1y[:, 1] - np.min(A1y[:, 1]), label='A')
plt.plot(B1y[:, 0] - 1000, B1y[:, 1] - np.min(B1y[:, 1]), label='B')
plt.plot(C1y[:, 0] - 1500, C1y[:, 1] - np.min(C1y[:, 1]), label='C')

plt.legend()

plt.grid()
plt.show()
