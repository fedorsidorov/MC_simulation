import numpy as np
import matplotlib.pyplot as plt

# %% No DEBER
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/No DEBER/1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/No DEBER/1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/No DEBER/1_3.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))

major_ticks = np.arange(0, 1001, 100)
minor_ticks = np.arange(0, 1001, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.show()

# %% 365
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/356/C_height_1/C1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/356/C_height_1/C1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/356/C_height_1/C1_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/356/C_height_1/C1_4.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

major_ticks = np.arange(0, 1001, 100)
minor_ticks = np.arange(0, 1001, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.show()

# %% 365 again
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/height_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/height_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/height_3.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))

major_ticks = np.arange(0, 701, 100)
minor_ticks = np.arange(0, 701, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.show()

# %% 357
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/height/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/height/1d.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/height/2a.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/height/2d.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/height/6a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/height/6d.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/height/8a.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/height/8d.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

major_ticks = np.arange(0, 701, 100)
minor_ticks = np.arange(0, 701, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('y, nm')

plt.show()

# %% 359
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/height/1a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/height/1d.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/height/A1_a.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/height/A1_d.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

major_ticks = np.arange(0, 701, 100)
minor_ticks = np.arange(0, 701, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('y, nm')

plt.show()

# %% 360
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/CD_height/CD_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/CD_height/CD_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/CD_height/CD_3.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))

major_ticks = np.arange(0, 701, 100)
minor_ticks = np.arange(0, 701, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('y, nm')

plt.show()

# %% 361
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/361/CD_height/CD_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/361/CD_height/CD_2.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))

major_ticks = np.arange(0, 701, 100)
minor_ticks = np.arange(0, 701, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('y, nm')

plt.show()

