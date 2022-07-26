import numpy as np
import matplotlib.pyplot as plt

# %% D
pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D/1b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D/1c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D/1d.csv', delimiter=',', skiprows=5)

xx = pr_4[:, 0] - 15600
zz = pr_4[:, 1] - np.min(pr_4[:, 1])

inds = np.where(np.logical_and(
    xx >= -2000, xx <= 2000
))

xx = xx[inds]
zz = zz[inds]

plt.figure(dpi=300)

# plt.plot(pr_1[1180:, 0], pr_1[1180:, 1] - np.min(pr_1[1180:, 1]))
# plt.plot(pr_2[:, 0] - 1000, pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0] - 1500, pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0] - 1500, pr_4[:, 1] - np.min(pr_4[:, 1]))

plt.plot(xx, zz)
plt.xlim(-2000, 2000)

plt.grid()
plt.show()

# np.save('xx_357_y_slice_1.npy', xx)
# np.save('zz_357_y_slice_1.npy', zz)

# %% D2
pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D2/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D2/1b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D2/1c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D2/1d.csv', delimiter=',', skiprows=5)

xx = pr_4[:, 0] - 15600
zz = pr_4[:, 1] - np.min(pr_4[:, 1])

inds = np.where(np.logical_and(
    xx >= -2000, xx <= 2000
))

xx = xx[inds]
zz = zz[inds]

plt.figure(dpi=300)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

# plt.plot(xx, zz)
# plt.xlim(-2000, 2000)

plt.grid()
plt.show()

# %% D3
pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D3/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D3/1b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D3/1c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357y/slice/D3/1d.csv', delimiter=',', skiprows=5)

xx = pr_4[:, 0] - 13000 + 250
zz = pr_4[:, 1] - np.min(pr_4[:, 1])

inds = np.where(np.logical_and(
    xx >= -2000, xx <= 2000
))

xx = xx[inds]
zz = zz[inds]

# zz[53:] += 4

xx_old = np.load('notebooks/DEBER_simulation/exp_profiles/357/xx_357_y_slice_1.npy')
zz_old = np.load('notebooks/DEBER_simulation/exp_profiles/357/zz_357_y_slice_1.npy')

plt.figure(dpi=300)

# plt.plot(pr_1[:, 0] - 15700, pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0] - 16000, pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0] - 16000, pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0] - 16000, pr_4[:, 1] - np.min(pr_4[:, 1]))

plt.plot(xx, zz)
# plt.plot(xx_old, zz_old)
# plt.xlim(-2500, 2500)
# plt.xlim(-1500, 1500)

plt.grid()
plt.show()

# %
# np.save('notebooks/DEBER_simulation/exp_profiles/357/xx_357_y_slice_D3_4.npy', xx)
# np.save('notebooks/DEBER_simulation/exp_profiles/357/zz_357_y_slice_D3_4.npy', zz)



