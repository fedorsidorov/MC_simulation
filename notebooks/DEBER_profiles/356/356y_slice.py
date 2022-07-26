import numpy as np
import matplotlib.pyplot as plt

# %% A
A1y = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/A/1b.csv', delimiter=',', skiprows=5)
B1y = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/B/1b.csv', delimiter=',', skiprows=5)
C1y = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/C/1b.csv', delimiter=',', skiprows=5)

xx = C1y[:, 0] - 10000
zz = C1y[:, 1] - np.min(C1y[:, 1])

plt.figure(dpi=300)

plt.plot(A1y[:, 0], A1y[:, 1] - np.min(A1y[:, 1]) - 25, label='A')
plt.plot(B1y[:, 0] - 1000, B1y[:, 1] - np.min(B1y[:, 1]), label='B')
plt.plot(C1y[:, 0] - 1500, C1y[:, 1] - np.min(C1y[:, 1]), label='C')

# plt.plot(xx, zz)

plt.legend()

# plt.xlim(-2000, 2000)

plt.grid()
plt.show()
