import numpy as np
import matplotlib.pyplot as plt

# %% 1
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/1/1_no_dark_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/1/1_no_dark_2.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.show()

# %% 2
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/2/1c_dark_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/2/1c_dark_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/2/2c_bright_1.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/2/2c_bright_2.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/2/2c_dark_1.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/2/2c_dark_2.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]) + 150)
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]) + 150)
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlim(3000, 12000)

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.show()

# %% 3
# pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/3/1_4_1.csv', delimiter=',', skiprows=5)
# pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/3/1_center_1.csv', delimiter=',', skiprows=5)
# pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/3/3_1_1.csv', delimiter=',', skiprows=5)
# pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/3/3_1_2.csv', delimiter=',', skiprows=5)
# pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/3_center/c_dark_1.csv', delimiter=',', skiprows=5)
# pr_6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/3_corner_90/c_dark_1.csv', delimiter=',', skiprows=5)
# pr_7 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/3_corner_90/c1_light_1.csv', delimiter=',', skiprows=5)

pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_5.csv', delimiter=',', skiprows=5)


fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))
# plt.plot(pr_6[:, 0], pr_6[:, 1] - np.min(pr_6[:, 1]))
# plt.plot(pr_7[:, 0], pr_7[:, 1] - np.min(pr_7[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.xlim(10000, 20000)

plt.show()

# %% trapeze_1
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/3-4.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/4.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/c1_1.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/c1_2.csv', delimiter=',', skiprows=5)
pr_6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/cl1_1.csv', delimiter=',', skiprows=5)
pr_7 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/cl1_2.csv', delimiter=',', skiprows=5)
pr_8 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/cr1_1.csv', delimiter=',', skiprows=5)
pr_9 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/cr1_2.csv', delimiter=',', skiprows=5)
pr_10 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/cu1_1.csv', delimiter=',', skiprows=5)
pr_11 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_1/cu1_2.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

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
plt.plot(pr_11[:, 0], pr_11[:, 1] - np.min(pr_11[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.show()

# %% trapeze_2
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_2_angle/cd_dark_2_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_2_angle/cu_dark_1_1.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_2_angle/cu_light_1_1.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_2_angle/cu_light_2_1.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/trapeze_2_angle/cu_light_2_2.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
plt.plot(pr_2[:, 0] - 1000, pr_2[:, 1] - np.min(pr_2[:, 1]))
# plt.plot(pr_3[:, 0], pr_3[:, 1] - np.min(pr_3[:, 1]))
plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.show()

# %% slice_1
pr_1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_1.csv', delimiter=',', skiprows=5)
pr_2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_2.csv', delimiter=',', skiprows=5)
pr_3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_3.csv', delimiter=',', skiprows=5)
pr_4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_4.csv', delimiter=',', skiprows=5)
pr_5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/WET/slice_1/3_1_center_slice_5.csv', delimiter=',', skiprows=5)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)

# plt.plot(pr_1[:, 0], pr_1[:, 1] - np.min(pr_1[:, 1]))
# plt.plot(pr_2[:, 0], pr_2[:, 1] - np.min(pr_2[:, 1]))
plt.plot(pr_3[:, 0] / 1000 - 38.7, pr_3[:, 1] - np.min(pr_3[:, 1]) - 9)
# plt.plot(pr_4[:, 0], pr_4[:, 1] - np.min(pr_4[:, 1]))
# plt.plot(pr_5[:, 0], pr_5[:, 1] - np.min(pr_5[:, 1]))

major_ticks = np.arange(0, 301, 100)
minor_ticks = np.arange(0, 301, 25)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')

plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.xlim(-10, 10)
# plt.xlim(-4, 4)

plt.show()

# %%
# np.save('notebooks/DEBER_profiles/wet_slice/xx.npy', pr_3[:, 0] / 1000 - 38.7)
# np.save('notebooks/DEBER_profiles/wet_slice/zz.npy', pr_3[:, 1] - np.min(pr_3[:, 1]) - 9)


