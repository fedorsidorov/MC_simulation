import numpy as np
import matplotlib.pyplot as plt
import importlib
from _outdated import MC_classes_nm as mcc
# from mapping import mapping_viscosity_80nm as mm
from mapping import mapping_viscosity_900nm as mm
from functions import plot_functions as pf
from functions import e_matrix_functions as emf
import indexes as ind

emf = importlib.reload(emf)
mcc = importlib.reload(mcc)
ind = importlib.reload(ind)
mm = importlib.reload(mm)
pf = importlib.reload(pf)

# %%
r_beam = 0
E0 = 20e+3

xx = mm.x_centers_5nm
zz_vac = np.zeros(len(xx))
# zz_vac = np.ones(len(xx)) * ((1 + np.cos(xx * np.pi / (lx_cm / 2))) * d_PMMA) / 5

plt.figure(dpi=300)
plt.plot(xx, zz_vac)
plt.plot(xx, np.ones(len(xx)) * mm.d_PMMA)
plt.show()

# %%
n_primaries_in_file = 30

for i in range(1):

    print(i)

    structure = mcc.Structure(
            d_PMMA=mm.d_PMMA,
            xx=xx,
            zz_vac=zz_vac,
            ly=mm.ly)

    simulator = mcc.Simulator(
        structure=structure,
        n_electrons=n_primaries_in_file,
        E0_eV=E0,
        r_beam=r_beam
    )

    simulator.prepare_e_deque()
    simulator.start_simulation()

    e_DATA = simulator.get_total_history()
    e_DATA_PMMA = e_DATA[np.where(e_DATA[:, ind.e_DATA_layer_id_ind] == ind.PMMA_ind)]
    # e_DATA_PMMA_val = \
    #     e_DATA_PMMA[np.where(e_DATA_PMMA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

    # np.save('data/e_DATA_Pv_80nm/e_DATA_' + str(i) + '.npy', e_DATA_PMMA_val)

# %%
pf.plot_e_DATA(e_DATA_PMMA, mm.d_PMMA, xx, zz_vac)
# pf.plot_e_DATA(e_DATA_PMMA, mm.d_PMMA, xx, zz_vac, limits=[[500, 1000], [0, 80]])

# %%
tr_num = 27

now_e_DATA_PMMA = emf.get_e_id_e_DATA(e_DATA_PMMA, n_primaries_in_file, tr_num)

now_e_DATA_PMMA_val = now_e_DATA_PMMA[np.where(now_e_DATA_PMMA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind)]

pf.plot_e_DATA(now_e_DATA_PMMA_val, mm.d_PMMA, xx, zz_vac)
