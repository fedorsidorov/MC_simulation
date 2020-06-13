import matplotlib.pyplot as plt

import importlib

import numpy as np

import MC_classes_DEBER as mcd
import mapping_exp_80nm_Camscan as mapping

mcd = importlib.reload(mcd)
mapping = importlib.reload(mapping)


# %%
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

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


# %%
d_PMMA = 80e-7
xx = np.linspace(mapping.x_min, mapping.x_max, 1000) * 1e-7
zz_vac = np.ones(len(xx)) * mapping.z_max * 1e-7
ly = mapping.l_y * 1e-7
r_beam = 100e-7

E0 = 20e+3
n_electrons = 10

structure = mcd.Structure(d_PMMA, xx, zz_vac, ly)

simulator = mcd.Simulator(
    structure=structure,
    n_electrons=n_electrons,
    E0_eV=E0,
    r_beam=r_beam
)
simulator.prepare_e_deque()
simulator.start_simulation()

DATA = simulator.get_total_history()
# np.save('/Volumes/ELEMENTS/e_DATA/15_keV_80nm/DATA_test_50.npy', DATA)
# DATA_Pn = DATA[np.where(np.logical_and(DATA[:, 2] == 0, DATA[:, 3] != 0))]
# np.save('/Volumes/ELEMENTS/e_DATA/20_keV_80nm/DATA_Pn_' + str(i) + '.npy', DATA_Pn)
# print(str(i) + '-th DATA file is saved')

plot_DATA(DATA)
