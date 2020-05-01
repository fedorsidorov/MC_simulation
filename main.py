import importlib

import numpy as np

import MC_classes

MC_classes = importlib.reload(MC_classes)


if __name__ == '__main__':
    sim = MC_classes.Simulator(500e-7, 100, 20000)
    sim.prepare_e_deque()
    sim.start_simulation()

    history = sim.get_total_history()
    np.save('data/e_DATA/DATA_0_easy.npy', history)
    # history_Pn = history[np.where(np.logical_and(history[:, 2] == 0, history[:, 3] != 0))]
    # np.save('data/e_DATA/DATA_Pn_0.npy', history_Pn)
