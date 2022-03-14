import numpy as np
import matplotlib.pyplot as plt

# %% A
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/A/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/A/1b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/A/1c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/A/1d.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2a.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2b.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/A/2a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/A/2b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/A/2c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/A/2d.csv', delimiter=',', skiprows=5)
# pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2a.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2b.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))

# plt.xlim(0, 3e+4)

plt.grid()
plt.show()


# %% B
# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/B/1a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/B/1b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/B/1c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/B/1d.csv', delimiter=',', skiprows=5)
# # pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2a.csv', delimiter=',', skiprows=5)
# # pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2b.csv', delimiter=',', skiprows=5)

pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/B/2a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/B/2b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/B/2c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/B/2d.csv', delimiter=',', skiprows=5)
# pr_5 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2a.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2b.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
# plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
# plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))

# plt.xlim(0, 3e+4)

plt.grid()
plt.show()


# %% C
# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/1a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/1b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/1c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/1d.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2d.csv', delimiter=',', skiprows=5)

pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/3a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/3b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/3c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/3d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

# plt.xlim(0, 3e+4)

plt.grid()
plt.show()


# %% C2
pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C2/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C2/1b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C2/1c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C2/1d.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/2d.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/3a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/3b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/3c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/C/3d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

# plt.xlim(0, 3e+4)

plt.grid()
plt.show()


# %% height
# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/A/1a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/A/1b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/A/1c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/A/1d.csv', delimiter=',', skiprows=5)

pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/B/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/B/1b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/B/1c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/B/1d.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/A/1a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/A/1b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/A/1c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/height/A/1d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

# plt.xlim(0, 3e+4)

plt.grid()
plt.show()


# %% height
# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/A/1a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/A/1b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/A/1c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/A/1d.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/B/1a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/B/1b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/B/1c.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/B/1d.csv', delimiter=',', skiprows=5)

pr_1 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/C/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/C/1b.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/C/1c.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/356/356y/slice/C/1d.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

# plt.xlim(0, 3e+4)

plt.grid()
plt.show()




