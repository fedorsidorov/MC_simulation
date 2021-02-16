import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import indexes as ind

ind = importlib.reload(ind)


# %%
def plot_e_DATA(e_DATA, d_PMMA, xx, zz_vac, E_cut=5, proj='xz', limits=None):
    e_DATA_cut = e_DATA[np.where(e_DATA[:, ind.e_DATA_E_ind] > E_cut)]
    fig, ax = plt.subplots(dpi=300)

    for e_id in range(int(np.max(e_DATA_cut[:, ind.e_DATA_e_id_ind]))):
        inds = np.where(e_DATA_cut[:, ind.e_DATA_e_id_ind] == e_id)[0]
        # now_e_DATA_cut = e_DATA_cut[inds, :]
        if len(inds) == 0:
            continue
        if proj == 'xz':
            xx_ind = ind.e_DATA_x_ind
            yy_ind = ind.e_DATA_z_ind
        elif proj == 'yz':
            xx_ind = ind.e_DATA_y_ind
            yy_ind = ind.e_DATA_z_ind
        elif proj == 'xy':
            xx_ind = ind.e_DATA_x_ind
            yy_ind = ind.e_DATA_y_ind
        else:
            print('specify projection: \'xy\', \'yz\' or \'yz\'')
            return
        # ax.plot(now_e_DATA_cut[:, xx_ind], now_e_DATA_cut[:, yy_ind])
        # ax.plot(e_DATA_cut[inds, xx_ind], e_DATA_cut[inds, yy_ind], '.-', linewidth='1')
        ax.plot(e_DATA_cut[inds, xx_ind], e_DATA_cut[inds, yy_ind], '-', linewidth='1')

    if proj != 'xy' and d_PMMA != 0:
        points = np.linspace(-d_PMMA * 2, d_PMMA * 2, 100)
        ax.plot(points, np.ones(len(points)) * d_PMMA, 'k')
        plt.plot(xx, zz_vac, 'k')

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')

    if limits:
        plt.xlim(limits[0])
        plt.ylim(limits[1])

    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


# %%
def plot_chain(chain_arr, beg=0, end=-1):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(chain_arr[beg:end, 0], chain_arr[beg:end, 1], chain_arr[beg:end, 2], 'bo-')
    #    ax.plot(chain_arr[0:-1, 0], chain_arr[0:-1, 1], chain_arr[0:-1, 2], 'bo-')

    ax.set_xlabel('x, nm')
    ax.set_ylabel('y, nm')
    ax.set_zlabel('z, nm')
    plt.show()


def plot_many_chains(chain_list):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    for chain in chain_list:
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2])

    ax.plot(np.linspace(const_m.x_min, const_m.x_max, const_m.l_x), np.ones(const_m.l_x)*const_m.y_min,
            np.ones(const_m.l_x)*const_m.z_min, 'k')
    ax.plot(np.linspace(const_m.x_min, const_m.x_max, const_m.l_x), np.ones(const_m.l_x)*const_m.y_max,
            np.ones(const_m.l_x)*const_m.z_min, 'k')
    ax.plot(np.linspace(const_m.x_min, const_m.x_max, const_m.l_x), np.ones(const_m.l_x)*const_m.y_min,
            np.ones(const_m.l_x)*const_m.z_max, 'k')
    ax.plot(np.linspace(const_m.x_min, const_m.x_max, const_m.l_x), np.ones(const_m.l_x)*const_m.y_max,
            np.ones(const_m.l_x)*const_m.z_max, 'k')

    ax.plot(np.ones(const_m.l_y)*const_m.x_min, np.linspace(const_m.y_min, const_m.y_max, const_m.l_y),
            np.ones(const_m.l_y)*const_m.z_min, 'k')
    ax.plot(np.ones(const_m.l_y)*const_m.x_max, np.linspace(const_m.y_min, const_m.y_max, const_m.l_y),
            np.ones(const_m.l_y)*const_m.z_min, 'k')
    ax.plot(np.ones(const_m.l_y)*const_m.x_min, np.linspace(const_m.y_min, const_m.y_max, const_m.l_y),
            np.ones(const_m.l_y)*const_m.z_max, 'k')
    ax.plot(np.ones(const_m.l_y)*const_m.x_max, np.linspace(const_m.y_min, const_m.y_max, const_m.l_y),
            np.ones(const_m.l_y)*const_m.z_max, 'k')

    ax.plot(np.ones(const_m.l_z)*const_m.x_min, np.ones(const_m.l_z)*const_m.y_min,
            np.linspace(const_m.z_min, const_m.z_max, const_m.l_z), 'k')
    ax.plot(np.ones(const_m.l_z)*const_m.x_max, np.ones(const_m.l_z)*const_m.y_min,
            np.linspace(const_m.z_min, const_m.z_max, const_m.l_z), 'k')
    ax.plot(np.ones(const_m.l_z)*const_m.x_min, np.ones(const_m.l_z)*const_m.y_max,
            np.linspace(const_m.z_min, const_m.z_max, const_m.l_z), 'k')
    ax.plot(np.ones(const_m.l_z)*const_m.x_max, np.ones(const_m.l_z)*const_m.y_max,
            np.linspace(const_m.z_min, const_m.z_max, const_m.l_z), 'k')

    plt.xlim(const_m.x_min, const_m.x_max)
    plt.ylim(const_m.y_min, const_m.y_max)
    plt.title('Polymer chain simulation')
    ax.set_xlabel('x, nm')
    ax.set_ylabel('y, nm')
    ax.set_zlabel('z, nm')
    plt.show()

