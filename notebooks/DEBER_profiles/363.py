import numpy as np
import matplotlib.pyplot as plt

# %%
D1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/363/1/lower_1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/363/2/1.csv', delimiter=',', skiprows=5)
D3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/363/3/2.csv', delimiter=',', skiprows=5)
D4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/363/4/1.csv', delimiter=',', skiprows=5)
D5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/363/5/1.csv', delimiter=',', skiprows=5)
D6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/363/6/1.csv', delimiter=',', skiprows=5)
D7 = np.loadtxt('notebooks/DEBER_profiles/Fedor/363/7/1.csv', delimiter=',', skiprows=5)
D8 = np.loadtxt('notebooks/DEBER_profiles/Fedor/363/8/1.csv', delimiter=',', skiprows=5)
# D8 = np.loadtxt('notebooks/DEBER_profiles/3-5.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

# plt.plot(D1[:200, 0], D1[:200, 1] - np.min(D1[:200, 1]), label='D1')
# plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')
# plt.plot(D3[:, 0], D3[:, 1] - np.min(D3[:, 1]), label='D3')
plt.plot(D4[:, 0], D4[:, 1] - np.min(D4[:, 1]), label='D4')
# plt.plot(D5[:, 0], D5[:, 1] - np.min(D5[:, 1]), label='D5')
# plt.plot(D6[:, 0], D6[:, 1] - np.min(D6[:, 1]), label='D6')
# plt.plot(D7[:, 0], D7[:, 1] - np.min(D7[:, 1]), label='D7')
# plt.plot(D8[:, 0], D8[:, 1] - np.min(D8[:, 1]), label='D8')

plt.legend()

plt.xlim(0, 5000)

plt.grid()
plt.show()