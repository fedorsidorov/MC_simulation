import importlib

import numpy as np

import Simulator as Simulator

Simulator = importlib.reload(Simulator)


if __name__ == '__main__':
    sim = Simulator.Simulator(500, 1, 10000)
    sim.prepare_e_deque()
    sim.start_simulation()

    history = sim.get_total_history()

    np.save('data/e_DATA/DATA_0.npy', history)
