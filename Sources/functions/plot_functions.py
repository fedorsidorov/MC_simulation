import importlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import indexes as ind
ind = importlib.reload(ind)


# %%
def plot_e_DATA(DATA, d_PMMA=0, E_cut=5, proj='xz'):
    DATA_cut = DATA[np.where(DATA[:, ind.DATA_E_ind] > E_cut)]
    fig, ax = plt.subplots(dpi=300)

    for e_id in range(int(np.max(DATA_cut[:, ind.DATA_e_id_ind]))):
        inds = np.where(DATA_cut[:, ind.DATA_e_id_ind] == e_id)[0]
        # now_DATA_cut = DATA_cut[inds, :]
        if len(inds) == 0:
            continue
        if proj == 'xz':
            xx_ind = ind.DATA_x_ind
            yy_ind = ind.DATA_z_ind
        elif proj == 'yz':
            xx_ind = ind.DATA_y_ind
            yy_ind = ind.DATA_z_ind
        elif proj == 'xy':
            xx_ind = ind.DATA_x_ind
            yy_ind = ind.DATA_y_ind
        else:
            print('specify projection: \'xy\', \'yz\' or \'yz\'')
            return
        # ax.plot(now_DATA_cut[:, xx_ind], now_DATA_cut[:, yy_ind])
        ax.plot(DATA_cut[inds, xx_ind], DATA_cut[inds, yy_ind])

    if proj != 'xy' and d_PMMA != 0:
        points = np.linspace(-d_PMMA * 2, d_PMMA * 2, 100)
        ax.plot(points, np.zeros(len(points)), 'k')
        ax.plot(points, np.ones(len(points)) * d_PMMA, 'k')

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


# %%
def plot_chain(chain_arr, beg=0, end=-1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(chain_arr[beg:end, 0], chain_arr[beg:end, 1], chain_arr[beg:end, 2], 'bo-')
    #    ax.plot(chain_arr[0:-1, 0], chain_arr[0:-1, 1], chain_arr[0:-1, 2], 'bo-')

    ax.set_xlabel('x, nm')
    ax.set_ylabel('y, nm')
    ax.set_zlabel('z, nm')
