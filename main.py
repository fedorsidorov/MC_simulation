from SimClasses.Simulator import Simulator


if __name__ == '__main__':

    sim = Simulator(100, 1, 1000)
    sim.prepare_e_deque()
    sim.start_simulation()
