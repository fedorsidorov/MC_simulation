# %%
import numpy as np
import matplotlib.pyplot as plt

# import mapping_harris as _outdated
import mapping_exp_100_3 as mapping
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
    plt.xlim(-250, 250)
    plt.ylim(0, 500)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


# %%
# d_PMMA_cm = 100e-7
d_PMMA_cm = mapping.d_PMMA_cm
# ly = 100e-7
lx_cm = mapping.l_x * 1e-7
ly_cm = mapping.l_y * 1e-7
r_beam = 100e-7
E0 = 20e+3

xx = mapping.x_centers_2nm * 1e-7
# zz_vac = np.zeros(len(xx))
zz_vac = np.ones(len(xx)) * np.cos(xx * np.pi / (lx_cm / 20)) * d_PMMA_cm / 2

plt.figure(dpi=300)
plt.plot(xx*1e+7, zz_vac*1e+7)
plt.show()

# %%
structure = mcd.Structure(d_PMMA=d_PMMA_cm, xx=xx, zz_vac=zz_vac, ly=ly_cm)
# print(structure.ly)
simulator = mcd.Simulator(structure=structure, n_electrons=20, E0_eV=E0, r_beam=r_beam)
simulator.prepare_e_deque()
simulator.start_simulation()

e_DATA = simulator.get_total_history()
# np.save('data/e_DATA/DATA_test_10.npy', e_DATA)

# %%
# now_DATA = np.load('data/e_DATA/DATA_test_50.npy')
# e_DATA_P = e_DATA[np.where(e_DATA[:, 2] == 0)]

plot_DATA(e_DATA, 100, E_cut=10)

# %%
# now_DATA_Pn = np.load('data/e_DATA/DATA_test.npy')
# ans = np.max(now_DATA[:, 0])
# bns = np.max(now_DATA_Pn[:, 0])
#
# a_inds = np.where(now_DATA[:, 0] == ans)[0]
# b_inds = np.where(now_DATA_Pn[:, 0] == bns)[0]
#
# cns = now_DATA[a_inds[0], :]
# dns = now_DATA_Pn[b_inds[0], :]
#
# %%
# plot_DATA(now_DATA, 500, E_cut=0)
