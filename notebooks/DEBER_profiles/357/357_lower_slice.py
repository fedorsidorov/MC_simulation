import numpy as np
import matplotlib.pyplot as plt

# %% A
pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/A_a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/A_b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/B_a.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/B_b.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/B_c.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/C_a.csv', delimiter=',', skiprows=5)
pr_7 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/C_b.csv', delimiter=',', skiprows=5)
pr_8 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/C_d.csv', delimiter=',', skiprows=5)
pr_9 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/D_a.csv', delimiter=',', skiprows=5)
pr_10 = np.loadtxt('notebooks/DEBER_profiles/357/357/lower_slice/D_b.csv', delimiter=',', skiprows=5)

ind_3 = 1820
ind_4 = 1700

xx_3 = pr_3[:ind_3, 0] - 45000 - 1200 - 250
zz_3 = pr_3[:ind_3, 1] - np.min(pr_3[:ind_3, 1])

inds_3 = np.where(np.logical_and(
    xx_3 >= -2000, xx_3 <= 2000
))

xx_3 = xx_3[inds_3]
zz_3 = zz_3[inds_3]

xx_4 = pr_4[:ind_4, 0] - 45000 - 1200
zz_4 = pr_4[:ind_4, 1] - np.min(pr_4[:ind_4, 1])

inds_4 = np.where(np.logical_and(
    xx_4 >= -2000, xx_4 <= 2000
))

xx_4 = xx_4[inds_4]
zz_4 = zz_4[inds_4]

plt.figure(dpi=300)

# plt.plot(xx_3, zz_3)
# plt.plot(xx_4, zz_4)

# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:ind_3, 0], pr_3[:ind_3, 1] - np.min(pr_3[:ind_3, 1]))
plt.plot(pr_4[:ind_4, 0], pr_4[:ind_4, 1] - np.min(pr_4[:ind_4, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
# plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
# plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))
# plt.plot(pr_8[:, 0], pr_8[:, 1] - np.min(pr_8[:, 1]))
# plt.plot(pr_9[:, 0], pr_9[:, 1] - np.min(pr_9[:, 1]))
# plt.plot(pr_10[:, 0], pr_10[:, 1] - np.min(pr_10[:, 1]))

# plt.xlim(-2500, 2500)

plt.grid()
plt.show()

# %%
# np.save('xx_357_lower_slice_3.npy', xx_3)
# np.save('zz_357_lower_slice_3.npy', zz_3)

# np.save('xx_357_lower_slice_4.npy', xx_4)
# np.save('zz_357_lower_slice_4.npy', zz_4)


