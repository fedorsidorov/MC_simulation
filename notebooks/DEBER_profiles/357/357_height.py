import numpy as np
import matplotlib.pyplot as plt

# %% A
pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/2a.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/3a.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/5a.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/6a.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/8a.csv', delimiter=',', skiprows=5)
pr_7 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/1d.csv', delimiter=',', skiprows=5)
pr_8 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/2d.csv', delimiter=',', skiprows=5)
pr_9 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/3d.csv', delimiter=',', skiprows=5)
pr_10 = np.loadtxt('notebooks/DEBER_profiles/357/357/height/5d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))
plt.plot(pr_8[:, 0], pr_8[:, 1] - np.min(pr_8[:, 1]))
plt.plot(pr_9[:, 0], pr_9[:, 1] - np.min(pr_9[:, 1]))
plt.plot(pr_10[:, 0], pr_10[:, 1] - np.min(pr_10[:, 1]))

# plt.xlim(-2500, 2500)

plt.grid()
plt.show()

# %%
# np.save('xx_357_lower_slice_3.npy', xx_3)
# np.save('zz_357_lower_slice_3.npy', zz_3)

# np.save('xx_357_lower_slice_4.npy', xx_4)
# np.save('zz_357_lower_slice_4.npy', zz_4)


