import numpy as np
import matplotlib.pyplot as plt

# %%
SE_output = np.loadtxt('notebooks/SE/vlist_narrow.txt')
# SE_output = np.loadtxt('notebooks/SE/vlist_wide.txt')

plt.figure(dpi=300)
plt.plot(SE_output[:, 1], SE_output[:, 2], '.')
plt.xlim(-5, 5)
plt.ylim(0, 0.06)

plt.show()

# %% parse file
times = []
profiles = []
beg = -1

for i, line in enumerate(SE_output):
    if line[1] == line[2] == -100:
        now_time = line[0]
        times.append(now_time)
        profiles.append(SE_output[beg+1:i, 1:])
        beg = i

# %%
yy = np.load('notebooks/Leveder/2010_sim/yy.npy')
zz_0 = np.load('notebooks/Leveder/2010_sim/0.npy')
zz_100 = np.load('notebooks/Leveder/2010_sim/100.npy')
zz_200 = np.load('notebooks/Leveder/2010_sim/200.npy')
zz_500 = np.load('notebooks/Leveder/2010_sim/500.npy')
zz_1200 = np.load('notebooks/Leveder/2010_sim/1200.npy')

yy = yy - 1

plt.figure(dpi=300)

plt.plot(yy, zz_0 * 1e-3, 'o-', label='Leveder 0')
plt.plot(yy, zz_100 * 1e-3, 'o-', label='Leveder 100')
plt.plot(yy, zz_200 * 1e-3, 'o-', label='Leveder 200')
plt.plot(yy, zz_500 * 1e-3, 'o-', label='Leveder 500')
plt.plot(yy, zz_1200 * 1e-3, 'o-', label='Leveder 1200')

i0, i1, i2, i3 = 0, 15, 32, 85

plt.plot(profiles[i0][:, 0], profiles[i0][:, 1], '.', label='SE ' + str(times[i0]))
plt.plot(profiles[i1][:, 0], profiles[i1][:, 1], '.', label='SE ' + str(times[i1]))
plt.plot(profiles[i2][:, 0], profiles[i2][:, 1], '.', label='SE ' + str(times[i2]))
plt.plot(profiles[i3][:, 0], profiles[i3][:, 1], '.', label='SE ' + str(times[i3]))
# plt.plot(profiles[i4][:, 0], profiles[i4][:, 1], '.', label='SE 0.75')

plt.xlim(-1, 1)
plt.ylim(0, 0.06)
plt.grid()
plt.legend()
plt.show()

# %%
plt.figure(dpi=300)
plt.plot([1, 100, 200, 500], [0, 0.075, 0.16, 0.425], 'ro')
plt.grid()
plt.show()
