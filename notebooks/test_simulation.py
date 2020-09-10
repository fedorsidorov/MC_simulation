# %%
import numpy as np
import matplotlib.pyplot as plt

import mapping_harris as mapping
import MC_classes_DEBER as mcd


def plot_DATA(DATA, d_PMMA=0, E_cut=5):
    print('initial size =', len(DATA))
    DATA_cut = DATA[np.where(DATA[:, 9] > E_cut)]
    print('cut DATA size =', len(DATA_cut))
    fig, ax = plt.subplots(dpi=300)

    for tn in range(int(np.max(DATA_cut[:, 0]))):
        if len(np.where(DATA_cut[:, 0] == tn)[0]) == 0:
            continue
        now_DATA_cut = DATA_cut[np.where(DATA_cut[:, 0] == tn)]
        ax.plot(now_DATA_cut[:, 4], now_DATA_cut[:, 6])

    if d_PMMA != 0:
        points = np.linspace(-d_PMMA * 2, d_PMMA * 2, 100)
        ax.plot(points, np.zeros(len(points)), 'k')
        ax.plot(points, np.ones(len(points)) * d_PMMA, 'k')

    # ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    # ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    # plt.xlim(-50, 50)
    # plt.ylim(0, 50)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


# %%
d_PMMA_cm = 100e-7
ly = 1
r_beam = 0
E0 = 10e+3

xx = mapping.x_centers_2nm * 1e-7
zz_vac = np.zeros(len(xx))
# zz_vac = np.ones(len(xx)) * np.cos(xx * np.pi / 100e-7) * d_PMMA/2

structure = mcd.Structure(d_PMMA=d_PMMA_cm, xx=xx, zz_vac=zz_vac, ly=ly)

simulator = mcd.Simulator(structure=structure, n_electrons=10, E0_eV=E0, r_beam=r_beam)
simulator.prepare_e_deque()
simulator.start_simulation()

e_DATA = simulator.get_total_history()
# np.save('data/e_DATA/DATA_test_10.npy', e_DATA)

# %%
# now_DATA = np.load('data/e_DATA/DATA_test_50.npy')
e_DATA_P = e_DATA[np.where(e_DATA[:, 2] == 0)]
plot_DATA(e_DATA, 500, E_cut=10)

# %%
now_DATA_Pn = np.load('data/e_DATA/DATA_test.npy')
ans = np.max(now_DATA[:, 0])
bns = np.max(now_DATA_Pn[:, 0])

a_inds = np.where(now_DATA[:, 0] == ans)[0]
b_inds = np.where(now_DATA_Pn[:, 0] == bns)[0]

cns = now_DATA[a_inds[0], :]
dns = now_DATA_Pn[b_inds[0], :]

# %%
plot_DATA(now_DATA, 500, E_cut=0)
