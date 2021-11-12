import numpy as np
import matplotlib.pyplot as plt

# %% B
B1y = np.loadtxt('notebooks/DEBER_profiles/357y/B/1d.csv', delimiter=',', skiprows=5)
B2y = np.loadtxt('notebooks/DEBER_profiles/357y/B2/1c.csv', delimiter=',', skiprows=5)
B3y = np.loadtxt('notebooks/DEBER_profiles/357y/B3/1e.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(B1y[:, 0], B1y[:, 1] - np.min(B1y[:, 1]), label='B')
# plt.plot(B2y[:, 0] + 300, B2y[:, 1] + 10)
# plt.plot(B3y[:, 0] - 1300, B3y[:, 1] + 15)

plt.grid()
# plt.show()

# % C
C1y = np.loadtxt('notebooks/DEBER_profiles/357y/C/1c.csv', delimiter=',', skiprows=5)
C2y = np.loadtxt('notebooks/DEBER_profiles/357y/C2/1c.csv', delimiter=',', skiprows=5)
C3y = np.loadtxt('notebooks/DEBER_profiles/357y/C3/1c.csv', delimiter=',', skiprows=5)

# plt.figure(dpi=300)

plt.plot(C1y[:, 0], C1y[:, 1] - np.min(C1y[:, 1]), label='C')
# plt.plot(C2y[:, 0] + 300, C2y[:, 1] + 10)
# plt.plot(C3y[:, 0] - 1300, C3y[:, 1] + 15)

plt.grid()
# plt.show()

# % D
D1y = np.loadtxt('notebooks/DEBER_profiles/357y/D/1a.csv', delimiter=',', skiprows=5)
D2y = np.loadtxt('notebooks/DEBER_profiles/357y/D2/1b.csv', delimiter=',', skiprows=5)
D3y = np.loadtxt('notebooks/DEBER_profiles/357y/D3/1b.csv', delimiter=',', skiprows=5)

# plt.figure(dpi=300)

plt.plot(D1y[:, 0], D1y[:, 1] - np.min(D1y[:, 1]), label='D')
# plt.plot(D2y[:, 0] + 1300, D2y[:, 1] + 10)
# plt.plot(D3y[:, 0] - 1000, D3y[:, 1] + 15)

plt.legend()
plt.grid()
plt.show()
