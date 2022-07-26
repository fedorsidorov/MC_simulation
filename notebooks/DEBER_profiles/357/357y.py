import numpy as np
import matplotlib.pyplot as plt

# %% D
pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D1/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D1/1b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D1/1c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D1/1d.csv', delimiter=',', skiprows=5)

pr_5 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D2/1a.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D2/1b.csv', delimiter=',', skiprows=5)
pr_7 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D2/1c.csv', delimiter=',', skiprows=5)
pr_8 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D2/1d.csv', delimiter=',', skiprows=5)

pr_9 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D3/1a.csv', delimiter=',', skiprows=5)
pr_10 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D3/1b.csv', delimiter=',', skiprows=5)
pr_11 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D3/1c.csv', delimiter=',', skiprows=5)
pr_12 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D3/1d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))
plt.plot(pr_8[:, 0], pr_8[:, 1] - np.min(pr_8[:, 1]))

plt.plot(pr_9[:, 0], pr_9[:, 1] - np.min(pr_9[:, 1]))
plt.plot(pr_10[:, 0], pr_10[:, 1] - np.min(pr_10[:, 1]))
plt.plot(pr_11[:, 0], pr_11[:, 1] - np.min(pr_11[:, 1]))
plt.plot(pr_12[:, 0], pr_12[:, 1] - np.min(pr_12[:, 1]))

plt.xlim(0, 10000)
plt.grid()
plt.show()

# %% C - MAY BE
C1y = np.loadtxt('notebooks/DEBER_profiles/357/357y/C1/1c.csv', delimiter=',', skiprows=5)
C2y = np.loadtxt('notebooks/DEBER_profiles/357/357y/C2/1c.csv', delimiter=',', skiprows=5)
C3y = np.loadtxt('notebooks/DEBER_profiles/357/357y/C3/1c.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(C1y[:, 0], C1y[:, 1] - np.min(C1y[:, 1]), label='C')
plt.plot(C2y[:, 0] + 300, C2y[:, 1] + 10)
plt.plot(C3y[:, 0] - 1300, C3y[:, 1] + 15)

plt.grid()
plt.show()

# %% 15now21 - NO
pr_1 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D1/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D2/1b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/357/357y/D3/1b.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]), label='15now21')
plt.plot(pr_2[:, 0] + 1300, pr_2[:, 1] + 10)
plt.plot(pr_3[:, 0] - 1000, pr_3[:, 1] + 15)

plt.legend()
plt.grid()
plt.show()
