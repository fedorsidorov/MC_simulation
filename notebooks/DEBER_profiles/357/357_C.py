import numpy as np
import matplotlib.pyplot as plt

# %% C1 - YES
C1_a = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C1_a.csv', delimiter=',', skiprows=5)
C1_b = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C1_b.csv', delimiter=',', skiprows=5)
C1_c = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C1_c.csv', delimiter=',', skiprows=5)
C1_d = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C1_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(C1_a[:, 0], C1_a[:, 1])
plt.plot(C1_b[:, 0] + 250, C1_b[:, 1] + 3)
plt.plot(C1_c[:, 0] + 300, C1_c[:, 1] + 4)
plt.plot(C1_d[:, 0] + 450, C1_d[:, 1] - 1)
plt.show()

# %% C2 - YES
C2_a = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C2_a.csv', delimiter=',', skiprows=5)
C2_b = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C2_b.csv', delimiter=',', skiprows=5)
C2_c = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C2_c.csv', delimiter=',', skiprows=5)
C2_d = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C2_d.csv', delimiter=',', skiprows=5)

# plt.figure(dpi=300)
plt.plot(C2_a[:, 0], C2_a[:, 1])
plt.plot(C2_b[:, 0] - 150, C2_b[:, 1] - 2)
plt.plot(C2_c[:, 0] - 250, C2_c[:, 1] - 3)
plt.plot(C2_d[:, 0] - 350, C2_d[:, 1] - 4)
plt.show()

# %% C3 - MAY BE
C3_a = np.loadtxt('notebooks/DEBER_profiles/357/C/C3_a.csv', delimiter=',', skiprows=5)
C3_b = np.loadtxt('notebooks/DEBER_profiles/357/C/C3_b.csv', delimiter=',', skiprows=5)
C3_c = np.loadtxt('notebooks/DEBER_profiles/357/C/C3_c.csv', delimiter=',', skiprows=5)
C3_d = np.loadtxt('notebooks/DEBER_profiles/357/C/C3_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(C3_a[:, 0], C3_a[:, 1])
plt.plot(C3_b[:, 0] - 0, C3_b[:, 1] + 1)
plt.plot(C3_c[:, 0] - 0, C3_c[:, 1] + 2)
plt.plot(C3_d[:, 0] - 0, C3_d[:, 1] - 1)
plt.show()

#%% C4 - NO
C4_a = np.loadtxt('notebooks/DEBER_profiles/357/C/C4_a.csv', delimiter=',', skiprows=5)
C4_b = np.loadtxt('notebooks/DEBER_profiles/357/C/C4_b.csv', delimiter=',', skiprows=5)
C4_c = np.loadtxt('notebooks/DEBER_profiles/357/C/C4_c.csv', delimiter=',', skiprows=5)
C4_d = np.loadtxt('notebooks/DEBER_profiles/357/C/C4_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(C4_a[:, 0], C4_a[:, 1])
plt.plot(C4_b[:, 0] + 700, C4_b[:, 1] + 1)
plt.plot(C4_c[:, 0] + 700, C4_c[:, 1] + 7)
plt.plot(C4_d[:, 0] + 800, C4_d[:, 1] + 7)
plt.show()

# %% C5 - YES
C5_a = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C5_a.csv', delimiter=',', skiprows=5)
C5_b = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C5_b.csv', delimiter=',', skiprows=5)
C5_c = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C5_c.csv', delimiter=',', skiprows=5)
C5_d = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/C/C5_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(C5_a[:, 0], C5_a[:, 1])
plt.plot(C5_b[:, 0] - 50, C5_b[:, 1] + 3)
plt.plot(C5_c[:, 0] - 100, C5_c[:, 1] + 2)
plt.plot(C5_d[:, 0] - 100, C5_d[:, 1] + 1)
plt.show()

# %% C6 - MAY BE
C6_a = np.loadtxt('notebooks/DEBER_profiles/357/C/C6_a.csv', delimiter=',', skiprows=5)
C6_b = np.loadtxt('notebooks/DEBER_profiles/357/C/C6_b.csv', delimiter=',', skiprows=5)
C6_c = np.loadtxt('notebooks/DEBER_profiles/357/C/C6_c.csv', delimiter=',', skiprows=5)
C6_d = np.loadtxt('notebooks/DEBER_profiles/357/C/C6_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(C6_a[:, 0], C6_a[:, 1])
plt.plot(C6_b[:, 0] - 0, C6_b[:, 1] + 3)
plt.plot(C6_c[:, 0] - 0, C6_c[:, 1] + 5)
plt.plot(C6_d[:, 0] - 0, C6_d[:, 1] + 5)
plt.show()

# %% C7 - NO
C7_a = np.loadtxt('notebooks/DEBER_profiles/357/C/C7_a.csv', delimiter=',', skiprows=5)
C7_b = np.loadtxt('notebooks/DEBER_profiles/357/C/C7_b.csv', delimiter=',', skiprows=5)
C7_c = np.loadtxt('notebooks/DEBER_profiles/357/C/C7_c.csv', delimiter=',', skiprows=5)
C7_d = np.loadtxt('notebooks/DEBER_profiles/357/C/C7_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(C7_a[:, 0], C7_a[:, 1])
plt.plot(C7_b[:, 0] - 0, C7_b[:, 1] - 2)
plt.plot(C7_c[:, 0] - 0, C7_c[:, 1] - 6)
plt.plot(C7_d[:, 0] - 0, C7_d[:, 1] - 6)
plt.show()

# %% C8 - MAY BE
C8_a = np.loadtxt('notebooks/DEBER_profiles/357/C/C8_a.csv', delimiter=',', skiprows=5)
C8_b = np.loadtxt('notebooks/DEBER_profiles/357/C/C8_b.csv', delimiter=',', skiprows=5)
C8_c = np.loadtxt('notebooks/DEBER_profiles/357/C/C8_c.csv', delimiter=',', skiprows=5)
C8_d = np.loadtxt('notebooks/DEBER_profiles/357/C/C8_d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(C8_a[:, 0], C8_a[:, 1])
plt.plot(C8_b[:, 0] - 0, C8_b[:, 1] + 2)
plt.plot(C8_c[:, 0] - 0, C8_c[:, 1] + 2)
plt.plot(C8_d[:, 0] - 0, C8_d[:, 1] + 3)
plt.show()

