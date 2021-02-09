import numpy as np
import importlib
import matplotlib.pyplot as plt
from mapping import mapping_3p3um_80nm as mm

mm = importlib.reload(mm)

# %%
profile = np.loadtxt('data/DEBER_profiles/Camscan_80nm/Camscan_new.txt')
inds = range(40, 162)
xx = profile[inds, 0]
yy = profile[inds, 1]

xx = xx - np.average(xx)
k = (yy[-1] - yy[0]) / (xx[-1] - xx[0])
yy_corr = yy - xx * k
yy_corr += 50

# plt.figure(dpi=300)
font_size = 8

_, ax = plt.subplots(dpi=300)
fig = plt.gcf()
fig.set_size_inches(4, 3)

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)


plt.plot(xx, yy_corr, '-', label='эксп.')
# plt.show()

zz_vac_f_5 = np.load('notebooks/DEBER_simulation/_outdated/zz_vac_fourier_5.npy')
zz_vac_SE_5 = np.load('notebooks/DEBER_simulation/_outdated/zz_vac_SE_5.npy')

xx_SE = np.load('notebooks/DEBER_simulation/_outdated/xx_SE_5.npy')

# zz_surf = np.load('notebooks/DEBER_simulation/_outdated/zz_surface_500_1e+5.npy')
zz_surf = np.load('notebooks/DEBER_simulation/_outdated/zz_surface_600_1e+5_final.npy')

plt.plot(mm.x_centers_10nm, zz_surf, '--', label='модел.')

# plt.plot(mm.x_centers_10nm, 80 - zz_vac_f_5, '-', label='zip length = 1000, $\eta$ = 10$^6$ Pa s')
# plt.plot(xx_SE, 80 - zz_vac_SE_5, '-', label='zip length = 1000, Surface Evolver')

plt.grid()

plt.xlim(-1800, 1800)
plt.ylim(0, 100)

# plt.title('Camscan profile, dose = 90 nC/cm$^2$')
plt.xlabel('x, нм', fontsize=font_size)
plt.ylabel('z, нм', fontsize=font_size)
plt.legend()

# plt.show()
plt.savefig('profile.tiff', bbox_inches='tight')

