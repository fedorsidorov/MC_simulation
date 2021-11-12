import numpy as np
import matplotlib.pyplot as plt

# %% B1 - YES
B1_a = np.loadtxt('notebooks/DEBER_profiles/357/B/B1_a.csv', delimiter=',', skiprows=5)
B1_b = np.loadtxt('notebooks/DEBER_profiles/357/B/B1_b.csv', delimiter=',', skiprows=5)
B1_c = np.loadtxt('notebooks/DEBER_profiles/357/B/B1_c.csv', delimiter=',', skiprows=5)
B1_d = np.loadtxt('notebooks/DEBER_profiles/357/B/B1_d.csv', delimiter=',', skiprows=5)

B1y = np.loadtxt('notebooks/DEBER_profiles/357y/B/1d.csv', delimiter=',', skiprows=5)
B2y = np.loadtxt('notebooks/DEBER_profiles/357y/B2/1c.csv', delimiter=',', skiprows=5)
B3y = np.loadtxt('notebooks/DEBER_profiles/357y/B3/1a.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(B1_a[:, 0], B1_a[:, 1])
plt.plot(B1_b[:, 0] - 250, B1_b[:, 1] + 0)
plt.plot(B1_c[:, 0] - 600, B1_c[:, 1] + 3)
plt.plot(B1_d[:, 0] - 950, B1_d[:, 1] + 7)

# plt.plot(B1y[:, 0], B1y[:, 1])
# plt.plot(B2y[:, 0] + 300, B2y[:, 1] + 10)
# plt.plot(B3y[:, 0] - 200, B3y[:, 1] + 0)

plt.grid()
plt.show()

# %% B2 - NO
B2_a = np.loadtxt('notebooks/DEBER_profiles/357/B/B2_a.csv', delimiter=',', skiprows=5)
B2_b = np.loadtxt('notebooks/DEBER_profiles/357/B/B2_b.csv', delimiter=',', skiprows=5)
B2_c = np.loadtxt('notebooks/DEBER_profiles/357/B/B2_c.csv', delimiter=',', skiprows=5)
B2_d = np.loadtxt('notebooks/DEBER_profiles/357/B/B2_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(B2_a[:, 0], B2_a[:, 1])
plt.plot(B2_b[:, 0] - 200, B2_b[:, 1] - 2)
plt.plot(B2_c[:, 0] - 450, B2_c[:, 1] + 3)
plt.plot(B2_d[:, 0] - 750, B2_d[:, 1] + 2)
plt.show()

# %% B3 - YES
B3_a = np.loadtxt('notebooks/DEBER_profiles/357/B/B3_a.csv', delimiter=',', skiprows=5)
B3_b = np.loadtxt('notebooks/DEBER_profiles/357/B/B3_b.csv', delimiter=',', skiprows=5)
B3_c = np.loadtxt('notebooks/DEBER_profiles/357/B/B3_c.csv', delimiter=',', skiprows=5)
B3_d = np.loadtxt('notebooks/DEBER_profiles/357/B/B3_d.csv', delimiter=',', skiprows=5)

# plt.figure(dpi=300)
# plt.plot(B3_a[:, 0], B3_a[:, 1])
# plt.plot(B3_b[:, 0] - 250, B3_b[:, 1] + 1)
# plt.plot(B3_c[:, 0] - 600, B3_c[:, 1] + 2)
plt.plot(B3_d[:, 0] - 700, B3_d[:, 1] + 2)
# plt.grid()
# plt.show()

#%% B4 - HARD NO
# B4_a = np.loadtxt('notebooks/DEBER_profiles/357/B/B4_a.csv', delimiter=',', skiprows=5)
# B4_b = np.loadtxt('notebooks/DEBER_profiles/357/B/B4_b.csv', delimiter=',', skiprows=5)
# B4_c = np.loadtxt('notebooks/DEBER_profiles/357/B/B4_c.csv', delimiter=',', skiprows=5)
# B4_d = np.loadtxt('notebooks/DEBER_profiles/357/B/B4_d.csv', delimiter=',', skiprows=5)

# plt.figure(dpi=300)
# plt.plot(B4_a[:, 0], B4_a[:, 1])
# plt.plot(B4_b[:, 0] - 100, B4_b[:, 1] + 1)
# plt.plot(B4_c[:, 0] + 100, B4_c[:, 1] + 7)
# plt.plot(B4_d[:, 0] - 100, B4_d[:, 1] + 7)
# plt.show()

# %% B5 - NO
# B5_a = np.loadtxt('notebooks/DEBER_profiles/357/B/B5_a.csv', delimiter=',', skiprows=5)
# B5_b = np.loadtxt('notebooks/DEBER_profiles/357/B/B5_b.csv', delimiter=',', skiprows=5)
# B5_c = np.loadtxt('notebooks/DEBER_profiles/357/B/B5_c.csv', delimiter=',', skiprows=5)
# B5_d = np.loadtxt('notebooks/DEBER_profiles/357/B/B5_d.csv', delimiter=',', skiprows=5)
#
# plt.figure(dpi=300)
# plt.plot(B5_a[:, 0], B5_a[:, 1])
# plt.plot(B5_b[:, 0] - 200, B5_b[:, 1] + 1)
# plt.plot(B5_c[:, 0] - 400, B5_c[:, 1] + 2)
# plt.plot(B5_d[:, 0] - 400, B5_d[:, 1] - 1)
# plt.show()

# %% B6 - NO
# B6_a = np.loadtxt('notebooks/DEBER_profiles/357/B/B6_a.csv', delimiter=',', skiprows=5)
# B6_b = np.loadtxt('notebooks/DEBER_profiles/357/B/B6_b.csv', delimiter=',', skiprows=5)
# B6_c = np.loadtxt('notebooks/DEBER_profiles/357/B/B6_c.csv', delimiter=',', skiprows=5)
# B6_d = np.loadtxt('notebooks/DEBER_profiles/357/B/B6_d.csv', delimiter=',', skiprows=5)
#
# plt.figure(dpi=300)
# plt.plot(B6_a[:, 0], B6_a[:, 1])
# plt.plot(B6_b[:, 0] - 200, B6_b[:, 1] + 3)
# plt.plot(B6_c[:, 0] - 500, B6_c[:, 1] - 1)
# plt.plot(B6_d[:, 0] - 750, B6_d[:, 1] + 2)
# plt.show()

# %% B7 - NO
# B7_a = np.loadtxt('notebooks/DEBER_profiles/357/B/B7_a.csv', delimiter=',', skiprows=5)
# B7_b = np.loadtxt('notebooks/DEBER_profiles/357/B/B7_b.csv', delimiter=',', skiprows=5)
# B7_c = np.loadtxt('notebooks/DEBER_profiles/357/B/B7_c.csv', delimiter=',', skiprows=5)
# B7_d = np.loadtxt('notebooks/DEBER_profiles/357/B/B7_d.csv', delimiter=',', skiprows=5)
#
# plt.figure(dpi=300)
# plt.plot(B7_a[:, 0], B7_a[:, 1])
# plt.plot(B7_b[:, 0] - 400, B7_b[:, 1] - 5)
# plt.plot(B7_c[:, 0] - 600, B7_c[:, 1] - 6)
# plt.plot(B7_d[:, 0] - 750, B7_d[:, 1] - 6)
# plt.show()

# %% B8 - MAY BE
B8_a = np.loadtxt('notebooks/DEBER_profiles/357/B/B8_a.csv', delimiter=',', skiprows=5)
B8_b = np.loadtxt('notebooks/DEBER_profiles/357/B/B8_b.csv', delimiter=',', skiprows=5)
B8_c = np.loadtxt('notebooks/DEBER_profiles/357/B/B8_c.csv', delimiter=',', skiprows=5)
B8_d = np.loadtxt('notebooks/DEBER_profiles/357/B/B8_d.csv', delimiter=',', skiprows=5)

# plt.figure(dpi=300)
# plt.plot(B8_a[:, 0], B8_a[:, 1])
# plt.plot(B8_b[:, 0] - 250, B8_b[:, 1] - 1)
plt.plot(B8_c[:, 0] - 500, B8_c[:, 1] - 2)
# plt.plot(B8_d[:, 0] - 650, B8_d[:, 1] - 3)
plt.show()

