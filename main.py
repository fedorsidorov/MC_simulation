import importlib

import numpy as np

import SimClasses.Simulator as Simulator

Simulator = importlib.reload(Simulator)


if __name__ == '__main__':
    sim = Simulator.Simulator(100, 1, 1000)
    sim.prepare_e_deque()
    sim.start_simulation()

    history = sim.get_total_history()