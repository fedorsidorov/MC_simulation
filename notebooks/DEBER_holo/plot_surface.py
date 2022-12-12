import importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

zz_0 = np.load('notebooks/DEBER_simulation/exp_profiles/366/zz_366_zero.npy') + 75
# zz_0 = np.load('notebooks/DEBER_vary_params/REF_zz_bins_avg.npy')
zz = np.concatenate([zz_0, zz_0, zz_0, zz_0, zz_0])

xx_final = np.arange(xx_0[0] - 6000, xx_0[-1] + 6000, 10)
zz_final = mcf.lin_lin_interp(xx, zz)(xx_final)

# plt.figure(dpi=300)
# plt.plot(xx_final, zz_final)
# plt.show()

# %
X, Y = np.meshgrid(xx_final, yy)

Z = np.zeros(np.shape(X))

for i in range(len(Z)):
    Z[i, :] = zz_final

# fig = plt.figure(dpi=300, figsize=[4, 3])
fig = plt.figure(dpi=300, figsize=[12, 7])
ax_3d = fig.add_subplot(projection='3d')

ax_3d.plot_surface(X / 1000, Y / 1000, Z, rstride=1, cstride=1, vmin=np.min(Z), vmax=np.max(Z)+30,
                   cmap=plt.get_cmap('gray'))
# ax_3d.plot_surface(X / 1000, Y / 1000, Z, rstride=1, cstride=1, vmin=np.min(Z), vmax=np.max(Z)+30,
#                    cmap=plt.get_cmap('viridis'))
# ax_3d.plot_surface(X / 1000, Y / 1000, Z, rstride=1, cstride=1, vmin=np.min(Z), vmax=np.max(Z),
#                    cmap=plt.get_cmap('viridis'))

ax_3d.set_xlabel(r'$x$, мкм')
ax_3d.set_ylabel(r'$y$, мкм')
ax_3d.set_zlabel(r'$z$, нм')

# ax_3d.set_xlim(-8, 8)
# ax_3d.set_ylim(-4, 4)
ax_3d.set_zlim(0, 500)

# ax_3d.view_init(azim=110, elev=20)
# ax_3d.view_init(azim=93, elev=35)
ax_3d.view_init(azim=93, elev=20)
# ax_3d.view_init(azim=87, elev=20)

plt.savefig('366_87_20_grey_new.jpg', dpi=600, bbox_inches='tight')
# plt.show()
