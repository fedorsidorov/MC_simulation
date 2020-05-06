import importlib
import numpy as np
import constants_physics as cp
import constants_mapping as cm

cp = importlib.reload(cp)
cm = importlib.reload(cm)

# %%
deg_path = 'C-C2:4_C-C\':2'

e_matrix_dE = np.load('data/e_matrix/' + deg_path + '/e_matrix_dE.npy')
chain_lens_initial = np.load('data/Harris/prepared_chains_1/prepared_chain_lens.npy')
chain_lens_final = np.load('data/Harris/lens_final_' + deg_path + '.npy')

Mn0 = np.average(chain_lens_initial)
Mn = np.average(chain_lens_final * cp.u_MMA)

total_E_loss = np.sum(e_matrix_dE)
scission_probability = (1 / Mn - 1 / Mn0) * cp.u_MMA  # probability of chain scission
sheet_density = cp.rho_PMMA * cm.harris_d_PMMA  # sheet density, g/cm^2

G_scission = (scission_probability * sheet_density * cp.Na * (1 / cp.u_MMA)) /\
             (total_E_loss / cm.harris_square) * 100

print('G(S) =', G_scission)


# %% direct calculation
scission_matrix_exc = np.load('data/e_matrix/' + deg_path + '/e_matrix_val_exc_sci.npy')
scission_matrix_ion = np.load('data/e_matrix/' + deg_path + '/e_matrix_val_ion_sci.npy')
dE_matrix = np.load('data/e_matrix/' + deg_path + '/e_matrix_dE.npy')

scission_matrix = scission_matrix_exc + scission_matrix_ion

print('direct G(S) =', np.sum(scission_matrix) / np.sum(dE_matrix) * 100)
