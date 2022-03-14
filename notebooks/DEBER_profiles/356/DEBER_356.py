import numpy as np
import matplotlib.pyplot as plt

# %% A
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/1_2b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/1_2b.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/1_2c.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/2_1.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/2_2.csv', delimiter=',', skiprows=5)
pr_7 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/2_3.csv', delimiter=',', skiprows=5)
pr_8 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/3_1.csv', delimiter=',', skiprows=5)
pr_9 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/3_2.csv', delimiter=',', skiprows=5)
pr_10 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/3_3.csv', delimiter=',', skiprows=5)
pr_11 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/3_3b.csv', delimiter=',', skiprows=5)
pr_12 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/3_3c.csv', delimiter=',', skiprows=5)
pr_13 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/3_3d.csv', delimiter=',', skiprows=5)
pr_14 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/3_3x_1.csv', delimiter=',', skiprows=5)
pr_15 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/3_4a.csv', delimiter=',', skiprows=5)
pr_16 = np.loadtxt('notebooks/DEBER_profiles/356/DEBER_356/3_4b.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
# plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
# plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
# plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))
# plt.plot(pr_8[:, 0], pr_8[:, 1] - np.min(pr_8[:, 1]))
# plt.plot(pr_9[:, 0], pr_9[:, 1] - np.min(pr_9[:, 1]))
# plt.plot(pr_10[:, 0], pr_10[:, 1] - np.min(pr_10[:, 1]))
# plt.plot(pr_11[:, 0], pr_11[:, 1] - np.min(pr_11[:, 1]))
# plt.plot(pr_12[:, 0], pr_12[:, 1] - np.min(pr_12[:, 1]))
plt.plot(pr_13[:, 0], pr_13[:, 1] - np.min(pr_13[:, 1]))
plt.plot(pr_14[:, 0], pr_14[:, 1] - np.min(pr_14[:, 1]))
plt.plot(pr_15[:, 0], pr_15[:, 1] - np.min(pr_15[:, 1]))
plt.plot(pr_16[:, 0], pr_16[:, 1] - np.min(pr_16[:, 1]))

plt.xlim(0, 3e+4)

plt.grid()
plt.show()





