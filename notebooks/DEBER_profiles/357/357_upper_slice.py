import numpy as np
import matplotlib.pyplot as plt

# %% A
pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357/upper_slice/A_a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357/upper_slice/A_b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357/upper_slice/B_a.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357/upper_slice/B_b.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/357/357/upper_slice/C_a.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/357/357/upper_slice/C_b.csv', delimiter=',', skiprows=5)
pr_7 = np.loadtxt('notebooks/DEBER_profiles/357/357/upper_slice/D_a.csv', delimiter=',', skiprows=5)
pr_8 = np.loadtxt('notebooks/DEBER_profiles/357/357/upper_slice/D_b.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))
plt.plot(pr_8[:, 0], pr_8[:, 1] - np.min(pr_8[:, 1]))

plt.grid()
plt.show()
