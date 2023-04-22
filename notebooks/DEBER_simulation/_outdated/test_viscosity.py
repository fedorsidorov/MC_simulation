import importlib
import numpy as np
import matplotlib.pyplot as plt
from mapping import mapping_3um_500nm as mm
from functions import SE_functions as ef
import constants as const
from functions import MC_functions as mcf
from functions import reflow_functions as rf

const = importlib.reload(const)
mcf = importlib.reload(mcf)
ef = importlib.reload(ef)
mm = importlib.reload(mm)
rf = importlib.reload(rf)

# %%
tau = np.load('notebooks/Boyd_Schulz_Zimm/arrays/tau.npy')
Mn_150 = np.load('notebooks/Boyd_Schulz_Zimm/arrays/Mn_150.npy') * 100

# PMMA 950K
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)

# %%
val_matrix = np.zeros((len(mm.x_centers_50nm), len(mm.z_centers_50nm)))

for i in range(1, 25):
    val_matrix += np.load('/Volumes/Transcend/val_matrix/val_matrix_' + str(i) + '.npy')

scission_weight = 0.09  # 150 C - 0.088568

scission_matrix = np.zeros(np.shape(val_matrix), dtype=int)

# fix extra vacuum events
for ii, _ in enumerate(mm.x_centers_50nm):
    for kk, zz in enumerate(mm.z_centers_50nm):

        n_val = int(val_matrix[ii, kk])
        scissions = np.where(np.random.random(n_val) < scission_weight)[0]
        scission_matrix[ii, kk] = len(scissions)

# %%
plt.figure(dpi=300)
plt.imshow(scission_matrix.transpose())
plt.show()

# %%
bin_volume = mm.step_50nm * mm.ly * mm.step_50nm
bin_n_monomers = bin_volume / const.V_mon_nm3

exp_time = 24

tau_matrix = np.zeros(np.shape(val_matrix))
Mn_matrix = np.zeros(np.shape(val_matrix))
eta_matrix = np.zeros(np.shape(val_matrix))

for i in range(len(mm.x_centers_50nm)):
    for j in range(len(mm.z_centers_50nm)):

        now_k_s = scission_matrix[i, j] / exp_time / bin_n_monomers
        now_tau = y_0 * now_k_s * 100
        tau_matrix[i, j] = now_tau

        now_Mn = mcf.lin_log_interp(tau, Mn_150)(now_tau)
        Mn_matrix[i, j] = now_Mn

        now_eta = rf.get_viscosity_experiment_Mn(150, now_Mn, 3.4)
        eta_matrix[i, j] = now_eta

# %%
plt.figure(dpi=300)
plt.imshow(np.log10(eta_matrix).transpose())
plt.colorbar()
plt.show()

# %%
eta_array = np.average(eta_matrix, axis=1)

plt.figure(dpi=300)
plt.semilogy(mm.x_centers_50nm, eta_array)
plt.show()

# %%
mobs_array = rf.get_SE_mobility(eta_array)

plt.figure(dpi=300)
plt.semilogy(mm.x_centers_50nm, mobs_array)
plt.show()






