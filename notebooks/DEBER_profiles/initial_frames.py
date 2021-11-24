import numpy as np
import matplotlib.pyplot as plt

# %% 356
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/2_1.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/2_2.csv', delimiter=',', skiprows=5)
# pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/3_1.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/3_2.csv', delimiter=',', skiprows=5)

pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/slice_2_1.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/Initial_Dark/356/slice_2_3.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/1_1.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/1_2a.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/2_2.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/3_1.csv', delimiter=',', skiprows=5)
# pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/3_3c.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/3_4b.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/3_1.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/3_2.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/3_3.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/3_3b.csv', delimiter=',', skiprows=5)
# pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/3_3c.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/DEBER_356/3_4a.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

# plt.plot(pr_1[:, 0] + 200, pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0] + 800, pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0] + 1400, pr_3[:, 1] - np.min(pr_3[:, 1]))
# plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

# plt.xlim(0, 10000)

plt.show()

# %% 357
# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D_new/D1_1.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D_new/D1_2.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D_new/D1_3.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D_new/D1_4.csv', delimiter=',', skiprows=5)

pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D1_a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D2_a.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D3_a.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D4_a.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D5_a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D6_a.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D7_a.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D8_a.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/lower_slice/D_a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/lower_slice/D_b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/upper_slice/D_a.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/upper_slice/D_b.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.show()

# %% 359 - D1x
# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D1_a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D1_d.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D2_a.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D2_d.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/Dx/1a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/Dx/2a.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/Dx/3a.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/359/D1x_slice_1.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359y/slice/D1/D1_a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359y/slice/D1/D1_b.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359y/slice/D1/D1_c.csv', delimiter=',', skiprows=5)

pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D7_a.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D8_a.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D11_a.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D8_a.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.show()

# %% 360
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/D_dark_1/D1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/D_dark_1/D1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/D_dark_2/D2_1.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/D_dark_2/D2_2.csv', delimiter=',', skiprows=5)

pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/360/D1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/360/D1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/360/D1_silce_1.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/360/D1_silce_2.csv', delimiter=',', skiprows=5)

pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/360/D1_1.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))

major_ticks = np.arange(0, 601, 100)
minor_ticks = np.arange(0, 601, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

# plt.xlim(0, 10000)

plt.show()

# %% 361
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/361/D_dark_1/D1_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/361/D_dark_1/D1_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/361/D_dark_2/D2_1.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/361/D_dark_2/D2_2.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.xlim(0, 10000)

plt.show()

# %% 362
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/362/B4_slice_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/362/B4_slice_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/362/C1_slice_1.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/15 nov 2021/362/C4_slice_1.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/D_dark_1/D1_2.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/D_dark_2/D2_1.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/360/D_dark_2/D2_2.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D3_a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D4_a.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D5_a.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D6_a.csv', delimiter=',', skiprows=5)

# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D7_a.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D8_a.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/359/D/D11_a.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/357/D/D8_a.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.show()
