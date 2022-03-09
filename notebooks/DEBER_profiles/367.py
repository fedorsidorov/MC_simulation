import numpy as np
import matplotlib.pyplot as plt

# %%
D1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/367/1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/367/2.csv', delimiter=',', skiprows=5)
D3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/367/3.csv', delimiter=',', skiprows=5)
D4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/367/4.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(D1[:, 0], D1[:, 1] - np.min(D1[:, 1]), label='D1')
plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')
plt.plot(D3[:, 0], D3[:, 1] - np.min(D3[:, 1]), label='D3')
plt.plot(D4[:, 0], D4[:, 1] - np.min(D4[:, 1]), label='D4')

plt.legend()

plt.xlim(0, 10000)

plt.grid()
plt.show()

# %% slice_1
D1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/367/slice_1/1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/367/slice_1/2.csv', delimiter=',', skiprows=5)
D3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/367/slice_1/3.csv', delimiter=',', skiprows=5)
D4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/367/slice_1/4.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(D1[:, 0], D1[:, 1] - np.min(D1[:, 1]), label='D1')
plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')
plt.plot(D3[:, 0], D3[:, 1] - np.min(D3[:, 1]), label='D3')
plt.plot(D4[:, 0], D4[:, 1] - np.min(D4[:, 1]), label='D4')

plt.legend()

plt.xlim(10000, 20000)

plt.grid()
plt.show()



