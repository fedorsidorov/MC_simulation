import importlib
import numpy as np
import matplotlib.pyplot as plt
from mapping import mapping_3um_500nm as mm

mm = importlib.reload(mm)


# %%
zz_vac_bins =\
    np.load('/Volumes/Transcend/SIM_DEBER/150C_100s_figures/new_s300_z150_pl1.4/zz_vac_bins_62.npy')
zz_inner_centers =\
    np.load('/Volumes/Transcend/SIM_DEBER/150C_100s_figures/new_s300_z150_pl1.4/zz_inner_centers_62.npy')

d_PMMA = 500
xx_bins, xx_centers = mm.x_bins_100nm, mm.x_centers_100nm

xx_total = np.zeros(len(xx_bins) + len(xx_centers))
zz_total = np.zeros(len(xx_bins) + len(xx_centers))

xx_total[0] = xx_bins[0]
zz_total[0] = zz_vac_bins[0]

# xx_total[-1] = xx_bins[-1]
# zz_total[-1] = zz_vac_bins[-1]

for i in range(len(xx_centers)):
    xx_total[1 + 2*i] = xx_centers[i]
    xx_total[2 + 2*i] = xx_bins[i + 1]

    zz_total[1 + 2*i] = zz_inner_centers[i]
    zz_total[2 + 2*i] = zz_vac_bins[i + 1]


# %%
with plt.style.context(['science', 'grid', 'russian-font']):
    fig, ax = plt.subplots(dpi=600)

    ax.plot(xx_bins, d_PMMA - zz_vac_bins, label=r'PMMA layer')
    ax.plot([-1], [-1], 'r', label='cavities')
    # ax.plot(xx_total, d_PMMA - zz_total, label=r'SE surface')
    # ax.plot(xx_centers, d_PMMA - zz_inner_centers, label=r'inner')

    # ax.set(title=r'$\eta$ = ' + str(format(now_eta, '.1e')) + ' Pa$\cdot$s')
    ax.set(xlabel=r'x, nm')
    ax.set(ylabel=r'y, nm')
    ax.legend(fontsize=7, loc='lower right')

    plt.xlim(-1500, 1500)
    plt.ylim(0, 600)

    plt.savefig('figures_final/profile_0.jpg', dpi=600)
    plt.show()


