import importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from functions import MC_functions as mcf
from mpl_toolkits.mplot3d import Axes3D

mcf = importlib.reload(mcf)

font = {'size': 14}
matplotlib.rc('font', **font)


# %%
xx_0 = np.load('notebooks/DEBER_simulation/exp_profiles/366/xx_366_zero.npy')
# xx_0 = np.load('notebooks/DEBER_vary_params/REF_xx_bins.npy')
xx = np.concatenate([xx_0 - 6000, xx_0 - 3000, xx_0, xx_0 + 3000, xx_0 + 6000])

yy = np.arange(-5000, 5001, 100)

zz_0 = np.load('notebooks/DEBER_simulation/exp_profiles/366/zz_366_zero.npy') + 86
# zz_0 = np.load('notebooks/DEBER_vary_params/REF_zz_bins_avg.npy')
zz = np.concatenate([zz_0, zz_0, zz_0, zz_0, zz_0])

xx_final = np.arange(xx_0[0] - 6000, xx_0[-1] + 6000, 10)
zz_final = mcf.lin_lin_interp(xx, zz)(xx_final)

X, Y = np.meshgrid(xx_final, yy)
Z = np.zeros(np.shape(X))

for i in range(len(Z)):
    Z[i, :] = zz_final

fig = plt.figure(dpi=300, figsize=[12, 7])
ax_3d = fig.add_subplot(projection='3d')

mesh_step = 2

# ax_3d.plot_surface(X / 1000, Y / 1000, Z, rstride=mesh_step, cstride=mesh_step, vmin=np.min(Z)-50, vmax=np.max(Z)+50,
#                    cmap=plt.get_cmap('gray'))

# ax_3d.plot_surface(X / 1000, Y / 1000, Z, rstride=mesh_step, cstride=mesh_step, vmin=np.min(Z), vmax=np.max(Z)),
#                    cmap=plt.get_cmap('viridis'))

ax_3d.plot_surface(X / 1000, Y / 1000, Z, rstride=mesh_step, cstride=mesh_step, vmin=np.min(Z)+30, vmax=np.max(Z)+100,
#     cmap=cm.coolwarm)
    # cmap=cm.Spectral_r)
    # cmap=cm.Paired)
    cmap=cm.YlOrBr_r)

ax_3d.tick_params(axis='y', pad=10)
ax_3d.tick_params(axis='z', pad=15)

ax_3d.set_title(r'эксперимент', fontsize=14)
# ax_3d.set_title(r'моделирование', fontsize=14)
ax_3d.set_xlabel('\n' + r'$x$, мкм', linespacing=1.6)
ax_3d.set_ylabel('\n' + r'$y$, мкм', linespacing=3)
ax_3d.set_zlabel('\n' + r'$z$, нм', linespacing=5)

ax_3d.set_xlim(-8, 8)
ax_3d.set_ylim(-4, 4)
ax_3d.set_zlim(0, 400)

ax_3d.view_init(azim=100, elev=30)

plt.savefig('366_EXP_' + str(mesh_step) + '_orange_50_title.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('366_SIM_50_orange_' + str(mesh_step) + '_title.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('366_YLORBr_r_NEW_30_100_' + str(mesh_step) + '.jpg', dpi=600, bbox_inches='tight')
plt.show()
