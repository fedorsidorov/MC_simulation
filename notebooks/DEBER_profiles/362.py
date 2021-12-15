import numpy as np
import matplotlib.pyplot as plt

# %%
D1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/362/B4_slice_1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/362/B4_slice_2.csv', delimiter=',', skiprows=5)
D3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/362/C1_slice_1.csv', delimiter=',', skiprows=5)
D4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/362/C4_slice_1.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(D1[:200, 0], D1[:200, 1] - np.min(D1[:200, 1]), label='D1')
plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')
plt.plot(D3[:, 0], D3[:, 1] - np.min(D3[:, 1]), label='D3')
plt.plot(D4[:, 0], D4[:, 1] - np.min(D4[:, 1]), label='D4')

plt.legend()
plt.grid()
plt.show()