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

plt.figure(dpi=300)
# plt.plot(xx, yy_corr, 'o-', label='experiment')
# plt.show()

zz_vac_f_5 = np.load('notebooks/DEBER_simulation/zz_vac_fourier_5.npy')
zz_vac_SE_5 = np.load('notebooks/DEBER_simulation/zz_vac_SE_5.npy')

xx_SE = np.load('notebooks/DEBER_simulation/xx_SE_5.npy')

# zz_surf = np.load('notebooks/DEBER_simulation/zz_surface_500_1e+5.npy')
# zz_surf = np.load('notebooks/DEBER_simulation/zz_surface_600_1e+5_final.npy')
plt.plot(mm.x_centers_10nm, 80 - zz_vac_f_5, '-', label='zip length = 1000, $\eta$ = 10$^6$ Pa s')
plt.plot(xx_SE, 80 - zz_vac_SE_5, '-', label='zip length = 1000, Surface Evolver')

plt.grid()

# plt.xlim(-1700, 1700)
plt.ylim(60, 90)

plt.title('Camscan profile, dose = 90 nC/cm$^2$')
plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.legend()

# plt.show()
plt.savefig('SE_vs_fourier.png', dpi=300)

