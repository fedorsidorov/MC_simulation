import importlib
import matplotlib.pyplot as plt
import numpy as np
import mapping_harris as mapping
import MC_classes_DEBER as mcd

mcd = importlib.reload(mcd)
mapping = importlib.reload(mapping)


# %%
# d_PMMA = 500e-7
# ly = 1
# r_beam = 0
#
# E0 = 10e+3
#
# xx = mapping.x_centers_2nm * 1e-7
# zz_vac = np.ones(len(xx)) * np.cos(xx * np.pi / 100e-7) * d_PMMA/2

# plt.figure(dpi=300)
# plt.plot(xx, zz_vac)
# plt.show()

# %
if __name__ == '__main__':

    d_PMMA = 500e-7
    ly = 1
    r_beam = 0

    E0 = 10e+3

    xx = mapping.x_centers_2nm * 1e-7
    zz_vac = np.zeros(len(xx))
    # zz_vac = np.ones(len(xx)) * np.cos(xx * np.pi / 100e-7) * d_PMMA/2

    structure = mcd.Structure(
        d_PMMA=d_PMMA,
        xx=xx,
        zz_vac=zz_vac,
        ly=ly)

    simulator = mcd.Simulator(
        structure=structure,
        n_electrons=50,
        E0_eV=E0,
        r_beam=r_beam
    )
    simulator.prepare_e_deque()
    simulator.start_simulation()

    e_DATA = simulator.get_total_history()

    np.save('data/e_DATA/DATA_test_50.npy', e_DATA)
    # DATA_Pn = DATA[np.where(np.logical_and(DATA[:, 2] == 0, DATA[:, 3] != 0))]
    # np.save('/Volumes/ELEMENTS/e_DATA/20_keV_80nm/DATA_Pn_' + str(i) + '.npy', DATA_Pn)
    # print(str(i) + '-th DATA file is saved')
