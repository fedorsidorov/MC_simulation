import numpy as np
import matplotlib.pyplot as plt

# %% A1
A1_c = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/Ax/1c.csv', delimiter=',', skiprows=5)
B1_a = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/Bx/1a.csv', delimiter=',', skiprows=5)
C1_a = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/Cx/1a.csv', delimiter=',', skiprows=5)
D1_a = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/Dx/1a.csv', delimiter=',', skiprows=5)

# %%
plt.figure(dpi=300)
plt.plot(A1_c[:, 0], A1_c[:, 1] - np.min(A1_c[:, 1]), label='A')
plt.plot(B1_a[:, 0] + 600, B1_a[:, 1] - np.min(B1_a[:, 1]), label='B')
plt.plot(C1_a[:, 0] + 1400, C1_a[:, 1] - np.min(C1_a[:, 1]), label='C')
plt.plot(D1_a[:, 0] - 1200, D1_a[:, 1] - np.min(D1_a[:, 1]), label='D')

plt.legend()

plt.grid()
plt.show()

# %% A4 - HARD NO
# A4_a = np.loadtxt('notebooks/DEBER_profiles/357/A/A4_a.csv', delimiter=',', skiprows=5)
# A4_b = np.loadtxt('notebooks/DEBER_profiles/357/A/A4_b.csv', delimiter=',', skiprows=5)
# A4_c = np.loadtxt('notebooks/DEBER_profiles/357/A/A4_c.csv', delimiter=',', skiprows=5)
# A4_d = np.loadtxt('notebooks/DEBER_profiles/357/A/A4_d.csv', delimiter=',', skiprows=5)

# plt.figure(dpi=300)
# plt.plot(A4_a[:, 0], A4_a[:, 1])
# plt.plot(A4_b[:, 0] - 100, A4_b[:, 1] + 1)
# plt.plot(A4_c[:, 0] + 100, A4_c[:, 1] + 7)
# plt.plot(A4_d[:, 0] - 100, A4_d[:, 1] + 7)
# plt.show()

# %% A5
A5_a = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/A/A5_a.csv', delimiter=',', skiprows=5)
A5_b = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/A/A5_b.csv', delimiter=',', skiprows=5)
A5_c = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/A/A5_c.csv', delimiter=',', skiprows=5)
A5_d = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/A/A5_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(A5_a[:, 0], A5_a[:, 1])
plt.plot(A5_b[:, 0] - 200, A5_b[:, 1] + 0)
plt.plot(A5_c[:, 0] - 300, A5_c[:, 1] - 1)
plt.plot(A5_d[:, 0] - 400, A5_d[:, 1] - 1)
plt.show()

# %% A6 - NO
A6_a = np.loadtxt('notebooks/DEBER_profiles/357/A/A6_a.csv', delimiter=',', skiprows=5)
A6_b = np.loadtxt('notebooks/DEBER_profiles/357/A/A6_b.csv', delimiter=',', skiprows=5)
A6_c = np.loadtxt('notebooks/DEBER_profiles/357/A/A6_c.csv', delimiter=',', skiprows=5)
A6_d = np.loadtxt('notebooks/DEBER_profiles/357/A/A6_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(A6_a[:, 0], A6_a[:, 1])
plt.plot(A6_b[:, 0] - 200, A6_b[:, 1] + 0)
plt.plot(A6_c[:, 0] - 500, A6_c[:, 1] - 1)
plt.plot(A6_d[:, 0] - 600, A6_d[:, 1] - 1)
plt.show()

# %% A7 - NO
A7_a = np.loadtxt('notebooks/DEBER_profiles/357/A/A7_a.csv', delimiter=',', skiprows=5)
A7_b = np.loadtxt('notebooks/DEBER_profiles/357/A/A7_b.csv', delimiter=',', skiprows=5)
A7_c = np.loadtxt('notebooks/DEBER_profiles/357/A/A7_c.csv', delimiter=',', skiprows=5)
A7_d = np.loadtxt('notebooks/DEBER_profiles/357/A/A7_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(A7_a[:, 0], A7_a[:, 1])
plt.plot(A7_b[:, 0] - 200, A7_b[:, 1] + 0)
plt.plot(A7_c[:, 0] - 300, A7_c[:, 1] - 1)
plt.plot(A7_d[:, 0] - 400, A7_d[:, 1] - 1)
plt.show()

# %% A8 - NO
A8_a = np.loadtxt('notebooks/DEBER_profiles/357/A/A8_a.csv', delimiter=',', skiprows=5)
A8_b = np.loadtxt('notebooks/DEBER_profiles/357/A/A8_b.csv', delimiter=',', skiprows=5)
A8_c = np.loadtxt('notebooks/DEBER_profiles/357/A/A8_c.csv', delimiter=',', skiprows=5)
A8_d = np.loadtxt('notebooks/DEBER_profiles/357/A/A8_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(A8_a[:, 0], A8_a[:, 1])
plt.plot(A8_b[:, 0] - 200, A8_b[:, 1] + 0)
plt.plot(A8_c[:, 0] - 400, A8_c[:, 1] - 2)
plt.plot(A8_d[:, 0] - 600, A8_d[:, 1] - 3)
plt.show()

