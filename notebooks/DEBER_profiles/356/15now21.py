import numpy as np
import matplotlib.pyplot as plt

# %% A
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/15now21/D1_slice_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/15now21/D1_slice_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/15now21/D4_1.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/15now21/Dx_1.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

plt.grid()
plt.show()
