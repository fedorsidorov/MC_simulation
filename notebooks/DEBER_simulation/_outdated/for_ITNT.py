import numpy as np
import matplotlib.pyplot as plt
import importlib
from mapping import mapping_3um_500nm as mm

mm = importlib.reload(mm)

# plt.style.use(['science', 'grid'])


# %%
x_step, z_step = 100, 100
x_bins, z_bins = mm.x_bins_100nm, mm.z_bins_100nm
x_centers, z_centers = mm.x_centers_100nm, mm.z_centers_100nm

xx = x_bins

zz_PMMA = np.load('notebooks/DEBER_simulation/NEW_zip_len_Mn_100nm/new_zz_PMMA_final_43.npy')
xx_SE = np.load('notebooks/DEBER_simulation/NEW_zip_len_Mn_100nm/SE_profile_x_43.npy')
yy_SE = np.load('notebooks/DEBER_simulation/NEW_zip_len_Mn_100nm/SE_profile_y_43.npy')

plt.figure(dpi=300, figsize=(4, 3))
plt.plot(xx, zz_PMMA, color='tab:blue', label='поверхность ПММА')
plt.plot(xx_SE, yy_SE, color='tab:orange', label='поверхность для растекания')

plt.xlabel('x, нм')
plt.ylabel('y, нм')

plt.xlim(-1500, 1500)
plt.ylim(0, 600)

plt.legend()
plt.grid()
# plt.show()
plt.savefig('for_ITNT_2.jpg', dpi=300, bbox_inches='tight')


