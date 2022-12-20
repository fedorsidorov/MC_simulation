import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %% 130C 100s
# % cooling reflow
TT_our = np.array([130,
               129, 128, 127, 126, 125, 124, 123, 122, 121, 120,
               119, 118, 117, 116, 115, 114, 113, 112, 111, 110,
               109, 108, 107, 106, 105, 104, 103, 102, 101, 100,
               99, 98, 97, 96, 95, 94, 93, 92, 91, 90,
               89, 88, 87, 86, 85, 84, 83, 82, 81, 80
               ])

tt_our_0 = np.array([4,
               3, 3, 4, 4, 3, 3, 4, 4, 4, 4,
               3, 4, 4, 5, 4, 4, 4, 5, 4, 4,
               5, 5, 4, 6, 4, 5, 5, 5, 5, 5,
               6, 6, 5, 6, 6, 5, 6, 6, 6, 7,
               7, 6, 7, 6, 8, 7, 7, 6, 9, 9
               ])

tt_our = np.zeros(len(TT_our))

tt_our[0] = tt_our_0[0]

for i in range(1, len(tt_our_0)):
    tt_our[i] = tt_our[i - 1] + tt_our_0[i]

TT_cooling_0p1 = np.arange(80, 131, 0.1)[::-1]
TT_cooling_0p2 = np.arange(80, 131, 0.2)[::-1]
TT_cooling_0p5 = np.arange(80, 131, 0.5)[::-1]
TT_cooling_1 = np.arange(80, 131, 1)[::-1]
TT_cooling_2 = np.arange(80, 131, 2)[::-1]
TT_cooling_5 = np.arange(80, 131, 5)[::-1]
TT_cooling_10 = np.arange(80, 131, 10)[::-1]

tt_0p1 = np.array(range(len(TT_cooling_0p1)))
tt_0p2 = np.array(range(len(TT_cooling_0p2)))
tt_0p5 = np.array(range(len(TT_cooling_0p5)))
tt_1 = np.array(range(len(TT_cooling_1)))
tt_2 = np.array(range(len(TT_cooling_2)))
tt_5 = np.array(range(len(TT_cooling_5)))
tt_10 = np.array(range(len(TT_cooling_10)))

# %%
plt.figure(dpi=300)

plt.plot(tt_our, TT_our, label=r'эксперимент')
# plt.plot(tt_0p1, TT_cooling_0p1)
plt.plot(tt_0p2, TT_cooling_0p2)
# plt.plot(tt_0p5, TT_cooling_0p5)
# plt.plot(tt_1, TT_cooling_1)
# plt.plot(tt_2, TT_cooling_2)
# plt.plot(tt_5, TT_cooling_5)
# plt.plot(tt_10, TT_cooling_10)

plt.grid()
plt.show()

# %%
xx_REF = np.load('notebooks/DEBER_cooling/130С_100s_REF_xx_bins.npy')
zz_REF = np.load('notebooks/DEBER_cooling/130С_100s_REF_zz_bins_avg.npy')

path = '/Volumes/Transcend/SIM_DEBER/130C_100s_cooling/'

xx_bins = np.load(path + '0.1C_sec/xx_bins.npy')
xx_total = np.load(path + '0.1C_sec/xx_total.npy')

zz_bins_0p05 = np.load(path + '0.05C_sec/zz_vac_bins.npy')
zz_bins_0p1 = np.load(path + '0.1C_sec/zz_vac_bins.npy')
zz_bins_0p2 = np.load(path + '0.2C_sec/zz_vac_bins.npy')
zz_bins_0p3 = np.load(path + '0.3C_sec/zz_vac_bins.npy')
zz_bins_0p4 = np.load(path + '0.4C_sec/zz_vac_bins.npy')
zz_bins_0p5 = np.load(path + '0.5C_sec/zz_vac_bins.npy')
zz_bins_1 = np.load(path + '1C_sec/zz_vac_bins.npy')

zz_total_0p05 = np.load(path + '0.05C_sec/zz_total.npy')
zz_total_0p1 = np.load(path + '0.1C_sec/zz_total.npy')
zz_total_0p2 = np.load(path + '0.2C_sec/zz_total.npy')
zz_total_0p3 = np.load(path + '0.3C_sec/zz_total.npy')
zz_total_0p4 = np.load(path + '0.4C_sec/zz_total.npy')
zz_total_0p5 = np.load(path + '0.5C_sec/zz_total.npy')
zz_total_1 = np.load(path + '1C_sec/zz_total.npy')

# %%
plt.figure(dpi=300, figsize=[4, 3])

plt.plot(xx_total / 1000, np.ones(len(xx_total)) * 500, 'C3--', label=r'начальная поверхность')

plt.plot(xx_REF / 1000, zz_REF, '--k', label=r'эксперимент')

# plt.plot(xx_total / 1000, zz_total_0p05, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins_0p05, 'C3', label=r'поверхность ПММА')
# plt.title(label=r'0.05$^\circ$C/c', fontsize=14)

# plt.plot(xx_total / 1000, zz_total_0p1, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins_0p1, 'C3', label=r'поверхность ПММА')
# plt.title(label=r'0.1$^\circ$C/c', fontsize=14)

# plt.plot(xx_total / 1000, zz_total_0p2, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins_0p2, 'C3', label=r'поверхность ПММА')
# plt.title(label=r'0.2$^\circ$C/c', fontsize=14)

# plt.plot(xx_total / 1000, zz_total_0p3, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins_0p3, 'C3', label=r'поверхность ПММА')
# plt.title(label=r'0.3$^\circ$C/c', fontsize=14)

plt.plot(xx_total / 1000, zz_total_0p4, 'C0', label=r'поверхность для растекания')
plt.plot(xx_bins / 1000, zz_bins_0p4, 'C3', label=r'поверхность ПММА')
plt.title(label=r'0.4$^\circ$C/c', fontsize=14)

# plt.plot(xx_total / 1000, zz_total_0p5, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins_0p5, 'C3', label=r'поверхность ПММА')
# plt.title(label=r'0.5$^\circ$C/c', fontsize=14)

# plt.plot(xx_total / 1000, zz_total_1, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins_1, 'C3', label=r'поверхность ПММА')
# plt.title(label=r'1$^\circ$C/c', fontsize=14)

# plt.plot(xx_total / 1000, zz_total_2, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins_2, 'C3', label=r'поверхность ПММА')
# plt.title(label=r'2$^\circ$C/c', fontsize=14)

# plt.plot(xx_total / 1000, zz_total_5, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins_5, 'C3', label=r'поверхность ПММА')
# plt.title(label=r'5$^\circ$C/c', fontsize=14)

# plt.plot(xx_total / 1000, zz_total_10, 'C0', label=r'поверхность для растекания')
# plt.plot(xx_bins / 1000, zz_bins_10, 'C3', label=r'поверхность ПММА')
# plt.title(label=r'10$^\circ$C/c', fontsize=14)

plt.legend(fontsize=10, loc='upper right')

plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$z$, нм')

plt.xlim(-1.5, 1.5)
plt.ylim(0, 800)
plt.grid()

plt.savefig('cooling_0p4.jpg', dpi=300, bbox_inches='tight')
plt.show()
