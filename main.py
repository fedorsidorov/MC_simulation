import importlib

import numpy as np

import MC_classes

MC_classes = importlib.reload(MC_classes)

if __name__ == '__main__':
    d_PMMA = 500e-7  # Harris
    # d_PMMA = 100e-7  # Aktary
    E0 = 10e+3
    n_electrons = 100

    for i in range(1):
        sim = MC_classes.Simulator(
            d_PMMA=d_PMMA,
            n_electrons=n_electrons,
            E0_eV=E0
        )
        sim.prepare_e_deque()
        sim.start_simulation()

        DATA = sim.get_total_history()
        np.save('data/e_DATA/DATA_test.npy', DATA)
        # DATA_Pn = DATA[np.where(np.logical_and(DATA[:, 2] == 0, DATA[:, 3] != 0))]
        # np.save('data/e_DATA/Aktary/DATA_Pn_' + str(i) + '.npy', DATA_Pn)
        # print(str(i) + '-th DATA file is saved')
