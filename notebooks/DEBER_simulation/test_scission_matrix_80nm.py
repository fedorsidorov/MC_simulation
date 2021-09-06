import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_3p3um_80nm as mm
from functions import array_functions as af
from functions import MC_functions as mcf
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import reflow_functions as rf
from functions import SE_functions as ef
from functions import e_beam_MC as eb_MC

import indexes as ind

const = importlib.reload(const)
eb_MC = importlib.reload(eb_MC)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
af = importlib.reload(af)
ef = importlib.reload(ef)
sf = importlib.reload(sf)
rf = importlib.reload(rf)

# %%
tau_125 = np.load('notebooks/Boyd_kinetic_curves/arrays/tau.npy')
Mn_125 = np.load('notebooks/Boyd_kinetic_curves/arrays/Mn_125.npy') * 100
Mw_125 = np.load('notebooks/Boyd_kinetic_curves/arrays/Mw_125.npy') * 100

xx = mm.x_centers_20nm
zz_vac = np.load('zz_vac.npy')
Mn_array = np.load('Mn_array.npy')
total_tau_array = np.load('total_tau_array.npy')

plt.figure(dpi=300)
plt.semilogy(tau_125, Mn_125)

plt.xlim(0, 400)
plt.ylim(1e+2, 1e+6)

plt.xlabel('tau')
plt.ylabel('Mn')
plt.grid()

plt.show()

# %%
plt.figure(dpi=300)
plt.plot(xx, mm.d_PMMA - zz_vac)

plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.grid()

plt.ylim(0, 80)

plt.show()

# %%
plt.figure(dpi=300)

plt.semilogy(xx, Mn_array)
# plt.semilogy(xx, total_tau_array)

plt.xlabel('x, nm')
plt.ylabel('Mn')
plt.grid()

plt.show()

# %%
eta_array = np.zeros(len(Mn_array))
mobs_array = np.zeros(len(Mn_array))

for i in range(len(eta_array)):
    eta_array[i] = rf.get_viscosity_experiment_Mn(125, Mn_array[i], power=3.4)
    mobs_array[i] = rf.get_SE_mobility(eta_array[i])

plt.figure(dpi=300)
plt.semilogy(xx, mobs_array)

plt.xlabel('x, nm')
plt.ylabel('SE mobility')
plt.grid()

plt.show()

# %%
zz_evolver = mm.d_PMMA - zz_vac

xx_evolver_final = np.concatenate([[mm.x_bins_20nm[0]], mm.x_centers_20nm, [mm.x_bins_20nm[-1]]]) * 1e-3
zz_evolver_final = np.concatenate([[zz_evolver[0]], zz_evolver, [zz_evolver[-1]]]) * 1e-3
mobs_evolver_final = np.ones(len(xx_evolver_final)) * 2e-3

plt.figure(dpi=300)
plt.plot(xx_evolver_final, zz_evolver_final)

plt.xlabel('x, nm')
plt.ylabel('z, nm')
plt.grid()

plt.show()

file_full_path = '/Users/fedor/PycharmProjects/MC_simulation/notebooks/SE/datafile_DEBER_2021.fe'

ef.create_datafile_latest(
    yy=xx_evolver_final,
    zz=zz_evolver_final,
    width=mm.ly * 1e-3,
    mobs=mobs_evolver_final,
    path=file_full_path
)

# %%
scission_matrix_1 = np.load('scission_matrix_80nm_step.npy')

scission_array_1 = np.sum(scission_matrix_1, axis=1)

plt.figure(dpi=300)
plt.plot(xx, scission_array_1)

plt.xlabel('x, nm')
plt.ylabel('n scissions')
plt.grid()

plt.show()

# %% get Mn corrected
Mn_0 = 2e+5
n_scissions_bin = 25

n_monomers_bin = mm.step_20nm * mm.ly * mm.d_PMMA / const.V_mon_nm3
# n_chains_bin = n_monomers_bin / (Mn_0 * 1e-2)
n_chains_bin = 1145

n_monomers_depol_bin = n_scissions_bin * 500

Mn_new = (1145 * Mn_0 + 12500) / (1145 + 12500)


