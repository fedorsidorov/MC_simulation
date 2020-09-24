import numpy as np
import matplotlib.pyplot as plt
import importlib
import MC_classes_nm as mcc
from mapping import mapping_viscosity as mm
from functions import plot_functions as pf

mcc = importlib.reload(mcc)
mm = importlib.reload(mm)
pf = importlib.reload(pf)

# %%
d_PMMA = mm.d_PMMA
lx_cm = mm.l_x
ly_cm = mm.l_y
r_beam = 100
E0 = 20e+3

xx = mm.x_centers_5nm
# zz_vac = np.zeros(len(xx))
zz_vac = np.ones(len(xx)) * ((1 + np.cos(xx * np.pi / (lx_cm / 2))) * d_PMMA) / 5

plt.figure(dpi=300)
plt.plot(xx, zz_vac)
plt.plot(xx, np.ones(len(xx)) * d_PMMA)
plt.show()

# %%
n_electrons = 50

for i in range(50):

    structure = mcc.Structure(
            d_PMMA=d_PMMA,
            xx=xx,
            zz_vac=zz_vac,
            ly=ly_cm)

    simulator = mcc.Simulator(
        structure=structure,
        n_electrons=n_electrons,
        E0_eV=E0,
        r_beam=r_beam
    )

    simulator.prepare_e_deque()
    simulator.start_simulation()

    e_DATA = simulator.get_total_history()
    np.save('data/e_DATA_test/e_DATA_' + str(i) + '.npy', e_DATA)

# %%
# now_DATA = np.load('data/e_DATA/DATA_test_50.npy')
# e_DATA_P = e_DATA[np.where(e_DATA[:, 2] == 0)]

pf.plot_e_DATA(e_DATA, d_PMMA, xx, zz_vac, limits=[[-200, 200], [-100, 200]])

# %%
# now_DATA_Pn = np.load('data/e_DATA/DATA_test.npy')
# ans = np.max(now_DATA[:, 0])
# bns = np.max(now_DATA_Pn[:, 0])
#
# a_inds = np.where(now_DATA[:, 0] == ans)[0]
# b_inds = np.where(now_DATA_Pn[:, 0] == bns)[0]
#
# cns = now_DATA[a_inds[0], :]
# dns = now_DATA_Pn[b_inds[0], :]
