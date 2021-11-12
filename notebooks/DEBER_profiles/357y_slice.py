import numpy as np
import matplotlib.pyplot as plt

# %% A
# A1y = np.loadtxt('notebooks/DEBER_profiles/357y/slice/A/1b.csv', delimiter=',', skiprows=5)
B1y = np.loadtxt('notebooks/DEBER_profiles/357y/slice/B/1d.csv', delimiter=',', skiprows=5)
C1y = np.loadtxt('notebooks/DEBER_profiles/357y/slice/C/1b.csv', delimiter=',', skiprows=5)
D1y = np.loadtxt('notebooks/DEBER_profiles/357y/slice/D/1b.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

# plt.plot(A1y[:, 0], A1y[:, 1] - np.min(A1y[:, 1]) - 25, label='A')
plt.plot(B1y[:, 0] - 1000, B1y[:, 1] - np.min(B1y[:, 1]), label='B')
plt.plot(C1y[:, 0] - 1500, C1y[:, 1] - np.min(C1y[:, 1]), label='C')
plt.plot(D1y[:, 0] - 1500, D1y[:, 1] - np.min(D1y[:, 1]), label='D')

plt.legend()

plt.grid()
plt.show()
