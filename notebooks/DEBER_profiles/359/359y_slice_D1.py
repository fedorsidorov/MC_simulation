import numpy as np
import matplotlib.pyplot as plt

# %%
pr_1 = np.loadtxt('notebooks/DEBER_profiles/359/359y/slice/D1/1_e.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/359/359y/slice/D1/D1_a.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/359/359y/slice/D1/D1_b.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/359/359y/slice/D1/D1_c.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/359/359y/slice/D1/D1_d.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/359/359y/slice/D1/D1_e.csv', delimiter=',', skiprows=5)

xx = pr_3[:, 0] - 30000 - 700 - 150
zz = pr_3[:, 1] - np.min(pr_3[:, 1])

# xx = pr_2[:, 0] - 30000 - 700 - 150
# zz = pr_2[:, 1] - np.min(pr_2[:, 1])

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))


# plt.plot(xx, zz)

# plt.xlim(-2000, 2000)
# plt.ylim(150, 250)

plt.grid()
plt.show()

# %%
# np.save('xx_359y_slise_D1.npy', xx)
# np.save('zz_359y_slise_D1.npy', zz)




