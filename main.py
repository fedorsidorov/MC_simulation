import importlib

import numpy as np

import MC_classes

MC_classes = importlib.reload(MC_classes)

if __name__ == '__main__':
    # d_PMMA = 500e-7  # Harris
    # d_PMMA = 100e-7  # combine_chains
    d_PMMA = 900e-7  # EXP
    # d_PMMA = 80e-7  # EXP
    # E0 = 10e+3
    E0 = 20e+3
    # E0 = 15e+3
    n_electrons = 100
    # n_electrons = 50

    for i in range(1000):
        sim = MC_classes.Simulator(
            d_PMMA=d_PMMA,
            n_electrons=n_electrons,
            E0_eV=E0
        )
        sim.prepare_e_deque()
        sim.start_simulation()

        DATA = sim.get_total_history()
        # np.save('/Volumes/ELEMENTS/e_DATA/15_keV_80nm/DATA_test_50.npy', DATA)
        DATA_Pn = DATA[np.where(np.logical_and(DATA[:, 2] == 0, DATA[:, 3] != 0))]
        np.save('/Volumes/ELEMENTS/e_DATA/20_keV_80nm/DATA_Pn_' + str(i) + '.npy', DATA_Pn)
        print(str(i) + '-th DATA file is saved')
