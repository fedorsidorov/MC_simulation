import numpy as np
import matplotlib.pyplot as plt

# %% A
A1 = np.loadtxt('notebooks/DEBER_profiles/360y/A1/A1_a.csv', delimiter=',', skiprows=5)
A2 = np.loadtxt('notebooks/DEBER_profiles/360y/A2/A2_d.csv', delimiter=',', skiprows=5)
B0 = np.loadtxt('notebooks/DEBER_profiles/360y/B0/B0_e.csv', delimiter=',', skiprows=5)
B1 = np.loadtxt('notebooks/DEBER_profiles/360y/B1/B1_d.csv', delimiter=',', skiprows=5)
B2 = np.loadtxt('notebooks/DEBER_profiles/360y/B2/B2_b.csv', delimiter=',', skiprows=5)

C0 = np.loadtxt('notebooks/DEBER_profiles/360y/C0/C0_1.csv', delimiter=',', skiprows=5)
# C0 = np.loadtxt('notebooks/DEBER_profiles/360y/C1/C1_4.csv', delimiter=',', skiprows=5)

D0 = np.loadtxt('notebooks/DEBER_profiles/360y/D1/D1_2.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(A1[:, 0], A1[:, 1] - np.min(A1[:, 1]), label='A1')
# plt.plot(A2[:, 0], A2[:, 1] - np.min(A2[:, 1]), label='A2')

plt.plot(B0[:, 0], B0[:, 1] - np.min(B0[:, 1]), label='B0')
# plt.plot(B1[:, 0], B1[:, 1] - np.min(B1[:, 1]), label='B1')
# plt.plot(B2[:, 0], B2[:, 1] - np.min(B2[:, 1]), label='B2')

plt.plot(C0[:, 0], C0[:, 1] - np.min(C0[:, 1]), label='C0')

plt.plot(D0[:, 0], D0[:, 1] - np.min(D0[:, 1]), label='D0')

plt.legend()
plt.grid()
plt.show()
