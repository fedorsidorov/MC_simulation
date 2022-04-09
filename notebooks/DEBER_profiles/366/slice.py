import numpy as np
import matplotlib.pyplot as plt

# %% slice
D1 = np.loadtxt('notebooks/DEBER_profiles/366/366/slice/1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/366/366/slice/2.csv', delimiter=',', skiprows=5)

xx = D2[:, 0] - 65000 - 600
zz = D2[:, 1] - np.min(D2[:, 1])

plt.figure(dpi=300)

plt.plot(D1[:, 0], D1[:, 1] - np.min(D1[:, 1]), label='D1')
plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')
# plt.plot(xx, zz)

plt.xlim(50000, 70000)
# plt.xlim(-20000, 20000)

plt.grid()
plt.show()
