import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)

# %%
# xx_bins = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/prec/xx_bins_140.npy')
# xx_centers = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/prec/xx_centers_140.npy')
# xx_total = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/prec/xx_total_140.npy')
#
# zz_vac_bins = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/prec/zz_vac_bins_140.npy')
# zz_inner_centers = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/prec/zz_inner_centers_140.npy')
# zz_total = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/prec/zz_total_140.npy')

# xx_bins = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/try_0/xx_bins.npy')
# xx_centers = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/try_0/xx_centers.npy')
# xx_total = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/try_0/xx_total.npy')
#
# zz_vac_bins = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/try_0/zz_vac_bins.npy')
# zz_inner_centers = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/try_0/zz_inner_centers.npy')
# zz_total = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_final/try_0/zz_total.npy')

xx_bins = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_cooling/0.4C_sec/xx_bins.npy')
xx_centers = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_cooling/0.4C_sec/xx_centers.npy')
xx_total = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_cooling/0.4C_sec/xx_total.npy')

zz_vac_bins = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_cooling/0.4C_sec/zz_vac_bins.npy')
zz_inner_centers = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_cooling/0.4C_sec/zz_inner_centers.npy')
zz_total = np.load('/Volumes/Transcend/SIM_DEBER/130C_100s_cooling/0.4C_sec/zz_total.npy')

plt.figure(dpi=600, figsize=[4, 3])

# plt.plot(xx_bins / 1000, zz_vac_bins, label='поверхность ПММА')
plt.plot(xx_bins / 1000, zz_vac_bins, 'C3', label='поверхность ПММА')
# plt.plot(xx_bins / 1000, zz_vac_bins, '--C0')
# plt.plot([-1], [-1], 'C3', label='микрополости')
# plt.plot([-1], [-1], 'C0', label='микрополости')
plt.plot(xx_total / 1000, zz_total, 'C0', label='поверхность для растекания')

plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

# plt.xlim(-1500, 1500)
plt.xlim(-1.5, 1.5)
plt.ylim(0, 700)
# plt.ylim(0, 600)

# plt.legend(loc='lower right', fontsize=10)
plt.legend(loc='upper right', fontsize=10)
plt.grid()

# plt.savefig('reflow_fin_14.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('reflow_model_a_inv.jpg', dpi=600, bbox_inches='tight')
plt.savefig('reflow_model_b_inv.jpg', dpi=200, bbox_inches='tight')
plt.show()
