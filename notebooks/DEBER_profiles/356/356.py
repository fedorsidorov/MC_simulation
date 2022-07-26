import numpy as np
import matplotlib.pyplot as plt

# %% B1
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/B1/B1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/B1/B1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/B1/B1_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/B1/B1_4.csv', delimiter=',', skiprows=5)
# pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2a.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2b.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
# plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
# plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))

plt.xlim(0, 3e+4)

plt.grid()
plt.show()


# %% B1_slice_1?
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/B_slice_1?/S1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/B_slice_1?/S1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/B_slice_1?/S1_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/B_slice_1?/S1_4.csv', delimiter=',', skiprows=5)
# pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2a.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2b.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
# plt.plot(pr_5[:, 0], pr_5[:, 1s] - np.min(pr_5[:, 1]))
# plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))

# plt.xlim(2e+4, 5e+4)

plt.grid()
plt.show()


# %% C1
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/C1/C1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/C1/C1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/C1/C1_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/C1/C1_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356/C1/C1_5.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356/C1/C1_6.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))

plt.xlim(2e+4, 5e+4)

plt.grid()
plt.show()


# %% C2
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/C2/C2_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/C2/C2_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/C2/C2_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/C2/C2_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356/C2/C2_5.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356/C2/C2_6.csv', delimiter=',', skiprows=5)
pr_7 = np.loadtxt('notebooks/DEBER_profiles/356/356/C2/C2_7.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))

plt.xlim(2e+4, 5e+4)

plt.grid()
plt.show()


# %% C3
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_5.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_6.csv', delimiter=',', skiprows=5)
# pr_7 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_7.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
# plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))

plt.xlim(2e+4, 5e+4)

plt.grid()
plt.show()


# %% C_height_1
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_5.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_6.csv', delimiter=',', skiprows=5)
# pr_7 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_7.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
# plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
# plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))

plt.xlim(2e+4, 5e+4)

plt.grid()
plt.show()


# %% C_slice_1
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_1/S1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_1/S1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_1/S1_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_1/S1_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_1/S1_5.csv', delimiter=',', skiprows=5)

xx = pr_5[:, 0] - 45000 + 410
zz = pr_5[:, 1] - np.min(pr_5[:, 1])

inds = np.where(np.logical_and(
    xx >= -2000, xx <= 2000
))

xx = xx[inds]
zz = zz[inds]

plt.figure(dpi=300)
# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))

# plt.plot(xx, zz)

# plt.xlim(-2000, 2000)

plt.grid()
plt.show()


# %% C_slice_2
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_2/S2_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_2/S2_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_2/S2_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_2/S2_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_2/S2_5.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_6.csv', delimiter=',', skiprows=5)
# pr_7 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_7.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
# plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
# plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
# plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))

plt.xlim(2e+4, 5e+4)

plt.grid()
plt.show()


# %% C_slice_3
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_3/S1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_3/S1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_3/S1_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_3/S1_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_slice_3/S1_5.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_6.csv', delimiter=',', skiprows=5)
# pr_7 = np.loadtxt('notebooks/DEBER_profiles/356/356/C3/C3_7.csv', delimiter=',', skiprows=5)

xx = pr_1[:, 0] - 43000 - 500
zz = pr_1[:, 1] - np.min(pr_1[:, 1])

inds = np.where(np.logical_and(
    xx >= -2000, xx <= 2000
))

xx = xx[inds]
zz = zz[inds]

np.save('xx_C_slice_3.npy', xx)
np.save('zz_C_slice_3.npy', zz)

plt.figure(dpi=300)

plt.plot(xx, zz)

# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
# plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
# plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
# plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))

# plt.xlim(2e+4, 5e+4)

plt.grid()
plt.show()






