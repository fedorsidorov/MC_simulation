# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


DATA = np.load('data/e_DATA/10keV_500nm/DATA_test_50.npy')
E_cut = 10
d_PMMA = 500

font_size = 8

print('initial size =', len(DATA))
DATA_cut = DATA[np.where(DATA[:, 9] > E_cut)]
print('cut DATA size =', len(DATA_cut))

_, ax = plt.subplots(dpi=600)
fig = plt.gcf()
fig.set_size_inches(3, 3)

for tn in range(int(np.max(DATA_cut[:, 0]))):
    if len(np.where(DATA_cut[:, 0] == tn)[0]) == 0:
        continue
    now_DATA_cut = DATA_cut[np.where(DATA_cut[:, 0] == tn)]
    # ax.plot(now_DATA_cut[:, 4], now_DATA_cut[:, 6])
    plt.plot(now_DATA_cut[:, 4], now_DATA_cut[:, 6], linewidth=1)

if d_PMMA != 0:
    points = np.linspace(-d_PMMA * 2, d_PMMA * 2, 100)
    # ax.plot(points, np.zeros(len(points)), 'k')
    #     # ax.plot(points, np.ones(len(points)) * d_PMMA, 'k')
    plt.plot(points, np.zeros(len(points)), 'k')
    plt.plot(points, np.ones(len(points)) * d_PMMA, 'k')

# plt.text(-1000, 320, 'PMMA', fontsize=font_size)
plt.text(-1250, 320, 'ПММА', fontsize=font_size)
plt.text(-1250, 1300, 'Si', fontsize=font_size)

# ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
# ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

# plt.xlabel(r'$x$, nm', fontsize=font_size)
# plt.ylabel(r'$z$, nm', fontsize=font_size)
plt.xlabel(r'$x$, нм', fontsize=font_size)
plt.ylabel(r'$z$, нм', fontsize=font_size)

plt.xlim(-1500, 1500)
plt.ylim(0, 2000)
plt.gca().invert_yaxis()
plt.grid()
# plt.show()

# %%
plt.savefig('tracks.tiff', bbox_inches='tight', dpi=600)
