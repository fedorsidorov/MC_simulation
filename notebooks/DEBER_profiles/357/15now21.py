import numpy as np
import matplotlib.pyplot as plt

# %% D
pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/15now21/D1_slice_1.csv', delimiter=',', skiprows=5)

xx = pr_1[:, 0] - 8550
zz = pr_1[:, 1] - np.min(pr_1[:, 1])

inds = np.where(np.logical_and(
    xx >= -2000, xx <= 2000
))

xx = xx[inds]
zz = zz[inds]

# np.save('xx_C_slice_3.npy', xx)
# np.save('zz_C_slice_3.npy', zz)

plt.figure(dpi=300)
# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(xx, zz)
plt.ylim(0, 500)
plt.grid()
plt.show()
