import numpy as np
import matplotlib.pyplot as plt

# %% D1 - YES
D1_a = np.loadtxt('notebooks/DEBER_profiles/357/D/D1_a.csv', delimiter=',', skiprows=5)
D1_b = np.loadtxt('notebooks/DEBER_profiles/357/D/D1_b.csv', delimiter=',', skiprows=5)
D1_c = np.loadtxt('notebooks/DEBER_profiles/357/D/D1_c.csv', delimiter=',', skiprows=5)
D1_d = np.loadtxt('notebooks/DEBER_profiles/357/D/D1_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(D1_a[:, 0], D1_a[:, 1])
# plt.plot(D1_b[:, 0] - 100, D1_b[:, 1] + 1)
# plt.plot(D1_c[:, 0] - 100, D1_c[:, 1] + 1)
# plt.plot(D1_d[:, 0] - 100, D1_d[:, 1] + 1)
# plt.show()

# %% D2 - YES
D2_a = np.loadtxt('notebooks/DEBER_profiles/357/D/D2_a.csv', delimiter=',', skiprows=5)
D2_b = np.loadtxt('notebooks/DEBER_profiles/357/D/D2_b.csv', delimiter=',', skiprows=5)
D2_c = np.loadtxt('notebooks/DEBER_profiles/357/D/D2_c.csv', delimiter=',', skiprows=5)
D2_d = np.loadtxt('notebooks/DEBER_profiles/357/D/D2_d.csv', delimiter=',', skiprows=5)

# plt.figure(dpi=300)
plt.plot(D2_a[:, 0], D2_a[:, 1])
# plt.plot(D2_b[:, 0] - 0, D2_b[:, 1] - 4)
# plt.plot(D2_c[:, 0] - 0, D2_c[:, 1] - 1)
# plt.plot(D2_d[:, 0] - 0, D2_d[:, 1] - 0)
# plt.show()

# %% D3 - NO
D3_a = np.loadtxt('notebooks/DEBER_profiles/357/D/D3_a.csv', delimiter=',', skiprows=5)
D3_b = np.loadtxt('notebooks/DEBER_profiles/357/D/D3_b.csv', delimiter=',', skiprows=5)
D3_c = np.loadtxt('notebooks/DEBER_profiles/357/D/D3_c.csv', delimiter=',', skiprows=5)
D3_d = np.loadtxt('notebooks/DEBER_profiles/357/D/D3_d.csv', delimiter=',', skiprows=5)

# plt.figure(dpi=300)
# plt.plot(D3_a[:, 0], D3_a[:, 1] + 4)
# plt.plot(D3_b[:, 0] - 0, D3_b[:, 1] + 1)
# plt.plot(D3_c[:, 0] - 0, D3_c[:, 1] + 2)
plt.plot(D3_d[:, 0] - 0, D3_d[:, 1] - 1)
# plt.show()

#%% D4 - NO
D4_a = np.loadtxt('notebooks/DEBER_profiles/357/D/D4_a.csv', delimiter=',', skiprows=5)
D4_b = np.loadtxt('notebooks/DEBER_profiles/357/D/D4_b.csv', delimiter=',', skiprows=5)
D4_c = np.loadtxt('notebooks/DEBER_profiles/357/D/D4_c.csv', delimiter=',', skiprows=5)
D4_d = np.loadtxt('notebooks/DEBER_profiles/357/D/D4_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(D4_a[:, 0] + 100, D4_a[:, 1] + 3)
plt.plot(D4_b[:, 0] + 200, D4_b[:, 1] + 1)
plt.plot(D4_c[:, 0] + 200, D4_c[:, 1] + 2)
plt.plot(D4_d[:, 0] + 300, D4_d[:, 1] + 2)
plt.show()

# %% D5 - YES
D5_a = np.loadtxt('notebooks/DEBER_profiles/357/D/D5_a.csv', delimiter=',', skiprows=5)
D5_b = np.loadtxt('notebooks/DEBER_profiles/357/D/D5_b.csv', delimiter=',', skiprows=5)
D5_c = np.loadtxt('notebooks/DEBER_profiles/357/D/D5_c.csv', delimiter=',', skiprows=5)
D5_d = np.loadtxt('notebooks/DEBER_profiles/357/D/D5_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
# plt.plot(D5_a[:, 0], D5_a[:, 1])
# plt.plot(D5_b[:, 0] - 0, D5_b[:, 1] + 1)
# plt.plot(D5_c[:, 0] - 0, D5_c[:, 1] + 3)
plt.plot(D5_d[:, 0] + 100, D5_d[:, 1] + 1)
plt.show()

# %% D6 - MAY BE
D6_a = np.loadtxt('notebooks/DEBER_profiles/357/D/D6_a.csv', delimiter=',', skiprows=5)
D6_b = np.loadtxt('notebooks/DEBER_profiles/357/D/D6_b.csv', delimiter=',', skiprows=5)
D6_c = np.loadtxt('notebooks/DEBER_profiles/357/D/D6_c.csv', delimiter=',', skiprows=5)
D6_d = np.loadtxt('notebooks/DEBER_profiles/357/D/D6_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(D6_a[:, 0], D6_a[:, 1])
plt.plot(D6_b[:, 0] - 0, D6_b[:, 1] + 1)
plt.plot(D6_c[:, 0] - 100, D6_c[:, 1] + 0)
plt.plot(D6_d[:, 0] - 150, D6_d[:, 1] + 0)
plt.show()

# %% D7 - NO
D7_a = np.loadtxt('notebooks/DEBER_profiles/357/D/D7_a.csv', delimiter=',', skiprows=5)
D7_b = np.loadtxt('notebooks/DEBER_profiles/357/D/D7_b.csv', delimiter=',', skiprows=5)
D7_c = np.loadtxt('notebooks/DEBER_profiles/357/D/D7_c.csv', delimiter=',', skiprows=5)
D7_d = np.loadtxt('notebooks/DEBER_profiles/357/D/D7_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(D7_a[:, 0], D7_a[:, 1])
plt.plot(D7_b[:, 0] - 0, D7_b[:, 1] - 2)
plt.plot(D7_c[:, 0] - 0, D7_c[:, 1] - 6)
plt.plot(D7_d[:, 0] - 0, D7_d[:, 1] - 6)
plt.show()

# %% D8 - MAY BE
D8_a = np.loadtxt('notebooks/DEBER_profiles/357/D/D8_a.csv', delimiter=',', skiprows=5)
D8_b = np.loadtxt('notebooks/DEBER_profiles/357/D/D8_b.csv', delimiter=',', skiprows=5)
D8_c = np.loadtxt('notebooks/DEBER_profiles/357/D/D8_c.csv', delimiter=',', skiprows=5)
D8_d = np.loadtxt('notebooks/DEBER_profiles/357/D/D8_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(D8_a[:, 0], D8_a[:, 1] + 2)
plt.plot(D8_b[:, 0] - 0, D8_b[:, 1] + 2)
plt.plot(D8_c[:, 0] - 0, D8_c[:, 1] + 2)
plt.plot(D8_d[:, 0] + 100, D8_d[:, 1] + 3)
plt.show()

