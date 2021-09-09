import numpy as np
import matplotlib.pyplot as plt
import importlib
from _outdated import MC_classes_cm as mcc
from mapping._outdated import mapping_viscosity_80nm as mm
from functions import plot_functions as pf

mcc = importlib.reload(mcc)
mm = importlib.reload(mm)
pf = importlib.reload(pf)

# %%
# d_PMMA_cm = 100e-7
d_PMMA_cm = mm.d_PMMA_cm
# ly = 100e-7
lx_cm = mm.l_x * 1e-7
ly_cm = mm.l_y * 1e-7
r_beam = 100e-7
E0 = 20e+3

xx = mm.x_centers_5nm * 1e-7
# zz_vac = np.zeros(len(xx))
zz_vac = np.ones(len(xx)) * (1 - np.cos(xx * np.pi / (lx_cm / 2)) / 5) * d_PMMA_cm

plt.figure(dpi=300)
plt.plot(xx*1e+7, zz_vac*1e+7)
plt.show()

# %%
for i in range(100):

    n_electrons = 50

    structure = mcc.Structure(
            d_PMMA=d_PMMA_cm,
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

# pf.plot_DATA(e_DATA, 100, E_cut=10)

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
#
# %%
# plot_DATA(now_DATA, 500, E_cut=0)
