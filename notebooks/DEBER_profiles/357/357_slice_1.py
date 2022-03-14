import numpy as np
import matplotlib.pyplot as plt

# %% A
# pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/1a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/1b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/1c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/1d.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/2a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/2b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/2c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/2d.csv', delimiter=',', skiprows=5)

pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/3a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/3b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/3c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357/slice_1/3d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]) - 25)
plt.plot(pr_2[:, 0] - 1000, pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0] - 1500, pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0] - 1500, pr_4[:, 1] - np.min(pr_4[:, 1]))

plt.grid()
plt.show()
