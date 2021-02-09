import numpy as np
import matplotlib.pyplot as plt
import importlib
import MC_classes_nm as mcc
from mapping import mapping_viscosity_80nm as mm
from functions import plot_functions as pf
import indexes as ind

mcc = importlib.reload(mcc)
ind = importlib.reload(ind)
mm = importlib.reload(mm)
pf = importlib.reload(pf)

# %%
# E0_arr = [100, 200, 300, 400, 500, 600, 800, 1000]
# E0_arr = [800, 1000]
E0_arr = [200]

d_PMMA = 1e+7

xx = mm.x_centers_5nm
zz_vac = np.zeros(len(xx))

# %%
# n_primaries_in_file = 100
n_primaries_in_file = 1

# n_files = 100
n_files = 1

for E0 in E0_arr:

    for i in range(n_files):

        print('E0 =', E0, 'i =', i)

        structure = mcc.Structure(
                d_PMMA=d_PMMA,
                xx=xx,
                zz_vac=zz_vac,
                ly=mm.ly)

        simulator = mcc.Simulator(
            structure=structure,
            n_electrons=n_primaries_in_file,
            E0_eV=E0,
            r_beam_x=0,
            r_beam_y=0
        )

        simulator.prepare_e_deque()
        simulator.start_simulation()

        e_DATA = simulator.get_total_history()
        e_DATA_outer = e_DATA[np.where(np.logical_and(e_DATA[:, ind.e_DATA_z_ind] < 0,
                                                      e_DATA[:, ind.e_DATA_E_ind] > 0))]

        # np.save('data/2ndaries/0p07_0p1/' + str(int(E0)) + '/e_DATA_outer_' + str(i) + '.npy', e_DATA_outer)

# %%
pf.plot_e_DATA(e_DATA, 0, xx, zz_vac)


