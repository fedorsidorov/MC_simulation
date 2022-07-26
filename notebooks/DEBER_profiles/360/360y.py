import numpy as np
import matplotlib.pyplot as plt

# %% A
# A1 = np.loadtxt('notebooks/DEBER_profiles/360y/A1/A1_a.csv', delimiter=',', skiprows=5)
# A2 = np.loadtxt('notebooks/DEBER_profiles/360y/A2/A2_d.csv', delimiter=',', skiprows=5)
# B0 = np.loadtxt('notebooks/DEBER_profiles/360y/B0/B0_e.csv', delimiter=',', skiprows=5)
# B1 = np.loadtxt('notebooks/DEBER_profiles/360y/B1/B1_d.csv', delimiter=',', skiprows=5)
# B2 = np.loadtxt('notebooks/DEBER_profiles/360y/B2/B2_b.csv', delimiter=',', skiprows=5)

# C0 = np.loadtxt('notebooks/DEBER_profiles/360y/C0/C0_1.csv', delimiter=',', skiprows=5)
# C0 = np.loadtxt('notebooks/DEBER_profiles/360y/C1/C1_4.csv', delimiter=',', skiprows=5)

D0 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_dark_1/D1_1.csv', delimiter=',', skiprows=5)
D1 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_dark_1/D1_2.csv', delimiter=',', skiprows=5)

D3 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_dark_2/D2_1.csv', delimiter=',', skiprows=5)
D4 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_dark_2/D2_2.csv', delimiter=',', skiprows=5)

D5 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_dark_3/D3_1.csv', delimiter=',', skiprows=5)
D6 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_dark_3/D3_2.csv', delimiter=',', skiprows=5)

C1 = np.loadtxt('notebooks/DEBER_profiles/360/360/C_dark_1/C1_1.csv', delimiter=',', skiprows=5)
C2 = np.loadtxt('notebooks/DEBER_profiles/360/360/C_dark_2/C2_1.csv', delimiter=',', skiprows=5)
C3 = np.loadtxt('notebooks/DEBER_profiles/360/360/C_dark_3/C3_1.csv', delimiter=',', skiprows=5)

B1 = np.loadtxt('notebooks/DEBER_profiles/360/360/B_dark_1/B1_1.csv', delimiter=',', skiprows=5)
B2 = np.loadtxt('notebooks/DEBER_profiles/360/360/B_dark_2/B2_1.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

# plt.plot(D0[:, 0], D0[:, 1] - np.min(D0[:, 1]), label='D0')
plt.plot(D3[:, 0], D3[:, 1] - np.min(D3[:, 1]), label='D3')

# plt.plot(D5[:, 0], D5[:, 1] - np.min(D5[:, 1]), label='D5')

plt.plot(C2[:, 0], C2[:, 1] - np.min(C2[:, 1]), label='С2')
# plt.plot(C3[:, 0], C3[:, 1] - np.min(C3[:, 1]), label='С3')

plt.plot(B1[:, 0], B1[:, 1] - np.min(B1[:, 1]), label='B1')
# plt.plot(B2[:, 0], B2[:, 1] - np.min(B2[:, 1]), label='B2')

plt.legend()
plt.grid()
plt.show()
