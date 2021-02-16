import numpy as np
import matplotlib.pyplot as plt
import os
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
E0_arr = [100, 200, 300, 400, 500, 600]
# E0_arr = [800, 1000]
# E0_arr = [200]

d_PMMA = 1e+7

xx = mm.x_centers_5nm
zz_vac = np.zeros(len(xx))

# %%
# n_primaries_in_file = 1
n_primaries_in_file = 100

# n_files = 50
n_files = 6

model = '0p1_0p15_0eV_4p05'

for i in range(n_files):

    for E0 in E0_arr:

        print('i =', i, ', E0 =', E0)

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

        dest_dir = os.path.join('data/2ndaries', model, str(E0))

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        np.save(dest_dir + '/e_DATA_outer_' + str(i) + '.npy', e_DATA_outer)

# %%
pf.plot_e_DATA(e_DATA, 0, xx, zz_vac)


