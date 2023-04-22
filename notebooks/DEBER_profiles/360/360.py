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


# %%
pr_1 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_slice/D1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_slice/D1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_slice/D1_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/360/360/D_slice/D1_4.csv', delimiter=',', skiprows=5)

ind_1 = 900
ind_2 = 850
ind_3 = 780
ind_4 = 740

# xx = pr_1[:ind_1, 0] - 3000 - 920
# zz = pr_1[:ind_1, 1] - np.min(pr_1[:ind_1, 1])

# xx = pr_2[:ind_2, 0] - 3000 - 920
# zz = pr_2[:ind_2, 1] - np.min(pr_2[:ind_2, 1])

# xx = pr_3[:ind_3, 0] - 3000 - 920
# zz = pr_3[:ind_3, 1] - np.min(pr_3[:ind_3, 1])

xx = pr_4[:ind_4, 0] - 3000 - 920
zz = pr_4[:ind_4, 1] - np.min(pr_4[:ind_4, 1])

inds = np.where(np.logical_and(
    xx >= -2000, xx <= 2000
))

xx = xx[inds]
zz = zz[inds]

plt.figure(dpi=300)

plt.plot(pr_1[:ind_1, 0], pr_1[:ind_1, 1] - np.min(pr_1[:ind_1, 1]))
plt.plot(pr_2[:ind_2, 0] + 2000, pr_2[:ind_2, 1] - np.min(pr_2[:ind_2, 1]))
plt.plot(pr_3[:ind_3, 0] + 4350, pr_3[:ind_3, 1] - np.min(pr_3[:ind_3, 1]))
plt.plot(pr_4[:ind_4, 0] + 5900, pr_4[:ind_4, 1] - np.min(pr_4[:ind_4, 1]))

# plt.plot(xx, zz)

# plt.xlim(20e+3, 35e+3)

plt.grid()
plt.show()

# %%
# xx_test = np.load('notebooks/DEBER_simulation/exp_profiles/360/xx_360.npy')
# zz_test = np.load('notebooks/DEBER_simulation/exp_profiles/360/zz_360.npy')

# plt.figure(dpi=300)
# plt.plot(xx_test, zz_test)
# plt.show()

# %% height
pr_1 = np.loadtxt('notebooks/DEBER_profiles/360/360/CD_height/CD_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/360/360/CD_height/CD_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/360/360/CD_height/CD_3.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))

plt.grid()
plt.show()



