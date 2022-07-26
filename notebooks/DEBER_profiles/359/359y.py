import numpy as np
import matplotlib.pyplot as plt

# %% A
A1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359y/A1/A1_b.csv', delimiter=',', skiprows=5)
A2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359y/A1/A1_f.csv', delimiter=',', skiprows=5)
A3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359y/A1_auto_zero/B1_g.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(A1[:, 0], A1[:, 1] - np.min(A1[:, 1]), label='A1')
plt.plot(A2[:, 0] + 900, A2[:, 1] - np.min(A2[:, 1]), label='A2')
plt.plot(A3[:, 0] - 700, A3[:, 1] - np.min(A3[:, 1]), label='A3')

plt.show()

# %%
B = np.loadtxt('notebooks/DEBER_profiles/Fedor/359y/B1/B1_h.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(B[:, 0] - 1000, B[:, 1] - np.min(B[:, 1]), label='B')

plt.legend()

plt.grid()
plt.show()

# %%
C = np.loadtxt('notebooks/DEBER_profiles/Fedor/359y/C1/C1_f.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(C[:, 0] + 300, C[:, 1] - np.min(C[:, 1]), label='C')

plt.legend()

plt.grid()
plt.show()

# %%
D = np.loadtxt('notebooks/DEBER_profiles/359/359y/D1x/D1_e.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(D[:, 0] + 100, D[:, 1] - np.min(D[:, 1]), label='15now21')

plt.legend()

plt.grid()
plt.show()
