import numpy as np
import matplotlib.pyplot as plt

# %%
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356/C_height_1/C1_5.csv', delimiter=',', skiprows=5)

pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/initial_dark/height_1.csv', delimiter=',', skiprows=5)
pr_7 = np.loadtxt('notebooks/DEBER_profiles/356/initial_dark/height_2.csv', delimiter=',', skiprows=5)
pr_8 = np.loadtxt('notebooks/DEBER_profiles/356/initial_dark/height_3.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))
plt.plot(pr_8[:, 0], pr_8[:, 1] - np.min(pr_8[:, 1]))

# plt.xlim(0, 3e+4)

plt.grid()
plt.show()
