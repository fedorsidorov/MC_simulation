import numpy as np
import matplotlib.pyplot as plt
import importlib
import MC_classes_pl as mcc
from mapping import mapping_viscosity_80nm as mm
from functions import plot_functions as pf
import indexes as ind

mcc = importlib.reload(mcc)
ind = importlib.reload(ind)
mm = importlib.reload(mm)
pf = importlib.reload(pf)

# %%
r_beam = 0

# E0 = 1e+2
# E0 = 1e+3
# E0 = 10e+3

# E0 = 10000

d_PMMA = 0

xx = mm.x_centers_5nm
zz_vac = np.zeros(len(xx))

n_primaries_in_file = 100
# n_primaries_in_file = 10

n_files = 100
# n_files = 1

for E0 in [100, 400, 1000, 4000]:

    for i in range(n_files):

        print(i)

        structure = mcc.Structure(
                d_PMMA=d_PMMA,
                xx=xx,
                zz_vac=zz_vac,
                ly=mm.ly)

        simulator = mcc.Simulator(
            structure=structure,
            n_electrons=n_primaries_in_file,
            E0_eV=E0,
            r_beam_x=r_beam,
            r_beam_y=r_beam
        )

        simulator.prepare_e_deque()
        simulator.start_simulation()

        e_DATA = simulator.get_total_history()

        np.save('data/MC_Si_pl/' + str(E0) + 'eV/e_DATA_' + str(i) + '.npy', e_DATA)

# %%
pf.plot_e_DATA(e_DATA, 0, xx, zz_vac)
