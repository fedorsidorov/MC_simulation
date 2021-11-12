import numpy as np
import matplotlib.pyplot as plt

# %% A
A1y = np.loadtxt('notebooks/DEBER_profiles/356y/A/1b.csv', delimiter=',', skiprows=5)
B1y = np.loadtxt('notebooks/DEBER_profiles/356y/B/1d.csv', delimiter=',', skiprows=5)
C1y = np.loadtxt('notebooks/DEBER_profiles/356y/C/1d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(A1y[:, 0], A1y[:, 1] - np.min(A1y[:, 1]), label='A')
plt.plot(B1y[:, 0] - 1000, B1y[:, 1] - np.min(B1y[:, 1]), label='B')
plt.plot(C1y[:, 0] - 1500, C1y[:, 1] - np.min(C1y[:, 1]), label='C')

plt.legend()

plt.grid()
plt.show()
