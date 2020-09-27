import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# %%
def plot_e_DATA(DATA, d_PMMA=0, E_cut=5, proj='xz'):
    DATA_cut = DATA[np.where(DATA[:, 9] > E_cut)]
    fig, ax = plt.subplots(dpi=300)

    for e_id in range(int(np.max(DATA_cut[:, 0]))):
        inds = np.where(DATA_cut[:, 0] == e_id)[0]
        # now_DATA_cut = DATA_cut[inds, :]
        if len(inds) == 0:
            continue
        if proj == 'xz':
            xx_ind = 4
            yy_ind = 6
        elif proj == 'yz':
            xx_ind = 5
            yy_ind = 6
        elif proj == 'xy':
            xx_ind = 4
            yy_ind = 5
        else:
            print('specify projection: \'xy\', \'yz\' or \'yz\'')
            return
        # ax.plot(now_DATA_cut[:, xx_ind], now_DATA_cut[:, yy_ind])
        ax.plot(DATA_cut[inds, xx_ind], DATA_cut[inds, yy_ind], '.-')

    if proj != 'xy' and d_PMMA != 0:
        points = np.linspace(-d_PMMA * 2, d_PMMA * 2, 100)
        ax.plot(points, np.zeros(len(points)), 'k')
        ax.plot(points, np.ones(len(points)) * d_PMMA, 'k')

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


# %%
e_DATA = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/e_DATA_test/DATA_test_50.npy')

# %%
plot_e_DATA(e_DATA, d_PMMA=80, E_cut=5, proj='xz')




