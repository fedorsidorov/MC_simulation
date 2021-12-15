import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# %%
# SE_n = np.loadtxt('notebooks/SE/vlist_narrow.txt')
SE_w = np.loadtxt('notebooks/SE/vlist_REF.txt')

# plt.figure(dpi=300)
# plt.plot(SE_output[:, 1], SE_output[:, 2], '.')
# plt.xlim(-5, 5)
# plt.ylim(0, 0.06)
# plt.show()

# %% parse file
times_n = []
times_w = []
profiles_n = []
profiles_w = []
beg = -1

# for i, line in enumerate(SE_n):
#     if line[1] == line[2] == -100:
#         now_time = line[0]
#         times_n.append(now_time)
#         profiles_n.append(SE_n[beg+1:i, 1:])
#         beg = i
#
# beg = -1

for i, line in enumerate(SE_w):
    if line[1] == line[2] == -100:
        now_time = line[0]
        times_w.append(now_time)
        profiles_w.append(SE_w[beg+1:i, 1:])
        beg = i

# %%
ind = 1

plt.figure(dpi=300)
plt.plot(profiles_w[ind][:, 0], profiles_w[ind][:, 1], '.')
plt.show()

# %%
yy = np.load('notebooks/Leveder/2010_sim/zz.npy')
zz_0 = np.load('notebooks/Leveder/2010_sim/0.npy')
zz_100 = np.load('notebooks/Leveder/2010_sim/100.npy')
zz_200 = np.load('notebooks/Leveder/2010_sim/200.npy')
zz_500 = np.load('notebooks/Leveder/2010_sim/500.npy')
zz_700 = np.load('notebooks/Leveder/2010_sim/700.npy')
zz_1000 = np.load('notebooks/Leveder/2010_sim/1000.npy')
zz_1200 = np.load('notebooks/Leveder/2010_sim/1200.npy')

yy = yy - 1

plt.figure(dpi=300)
plt.plot(yy, zz_0 * 1e-3, '--', label='Leveder 0')
plt.plot(yy, zz_100 * 1e-3, '--', label='Leveder 100')
plt.plot(yy, zz_200 * 1e-3, '--', label='Leveder 200')
plt.plot(yy, zz_500 * 1e-3, '--', label='Leveder 500')
plt.plot(yy, zz_700 * 1e-3, '--', label='Leveder 700')
# plt.plot(yy, zz_1000 * 1e-3, '--', label='Leveder 1000')
# plt.plot(yy, zz_1200 * 1e-3, '--', label='Leveder 1200')

i_0, i_100, i_200, i_500 = 0, 15, 32, 85
i_700, i_1000, i_1200 = 12, -1, -1

plt.plot(profiles_n[i_0][:, 0], profiles_n[i_0][:, 1], '.', label='SE ' + str(times_n[i_0]))
plt.plot(profiles_n[i_100][:, 0], profiles_n[i_100][:, 1], '.', label='SE ' + str(times_n[i_100]))
plt.plot(profiles_n[i_200][:, 0], profiles_n[i_200][:, 1], '.', label='SE ' + str(times_n[i_200]))
plt.plot(profiles_n[i_500][:, 0], profiles_n[i_500][:, 1], '.', label='SE ' + str(times_n[i_500]))
plt.plot(profiles_w[i_700][:, 0], profiles_w[i_700][:, 1], '.', label='SE ' + str(times_w[i_700]))
# plt.plot(profiles_w[i_1000][:, 0], profiles_w[i_1000][:, 1], '.', label='SE ' + str(times_w[i_1000]))
# plt.plot(profiles_w[i_1200][:, 0], profiles_w[i_1200][:, 1], '.', label='SE ' + str(times_w[i_1200]))

plt.xlim(-1, 1)
plt.ylim(0.02, 0.06)
plt.grid()
plt.legend()
plt.show()


# %%
def func(xx, alpha):
    return xx * alpha


tt_real = np.array((0, 100, 200, 500, 700))
tt_SE = np.array((0, 0.075, 0.16, 0.425, 0.6))

xx_f = np.linspace(-100, 800, 100)
alpha_fit = optimize.curve_fit(func, tt_real, tt_SE)[0]
yy_f = func(xx_f, alpha_fit)

plt.figure(dpi=300)
plt.plot(tt_real, tt_SE, 'ro')
plt.plot(xx_f, yy_f)

plt.grid()
plt.xlim(-100, 800)
plt.ylim(-0.1, 0.7)

plt.show()
