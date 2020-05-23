import importlib
import numpy as np
import constants as const
import mapping_harris as mapping
# import mapping_aktary as mapping

const = importlib.reload(const)
mapping = importlib.reload(mapping)

# %%
folder_name = 'Harris'
# folder_name = 'Aktary'
# deg_paths = 'C-C2:4'
deg_paths = 'C-C2:4_C-C\':2'
# deg_paths = 'C-C2:4_C-C\':2_C-C3:1'

# %%
e_matrix_E_dep = np.load('data/e_matrix/' + folder_name + '/' + deg_paths + '/e_matrix_E_dep.npy')
# chain_lens_initial = np.load('data/chains/Harris/lens_initial.npy')
chain_lens_initial = np.load('/Volumes/ELEMENTS/PyCharm_may/prepared_chains/Harris/chain_lens.npy')
# chain_lens_initial = np.load('data/chains/' + folder_name + '/prepared_chains/prepared_chain_lens.npy')
# chain_lens_final = np.load('/Volumes/ELEMENTS/PyCharm_may/chains/Harris/lens_final_' + deg_paths + '.npy')
chain_lens_final = np.load('data/exposed_chains/Harris/harris_lens_final_4+2_2nm.npy')

Mn = np.average(chain_lens_initial) * const.u_MMA
Mf = np.average(chain_lens_final) * const.u_MMA

total_E_loss = np.sum(e_matrix_E_dep)
G_scission = (Mn/Mf - 1) * const.rho_PMMA * const.Na / (total_E_loss / mapping.volume_cm3 * Mn) * 100

print('G(S) =', G_scission)


# %% direct calculation
scission_matrix_exc = np.load('data/e_matrix/' + folder_name + '/' + deg_paths + '/e_matrix_val_exc_sci.npy')
scission_matrix_ion = np.load('data/e_matrix/' + folder_name + '/' + deg_paths + '/e_matrix_val_ion_sci.npy')

scission_matrix = scission_matrix_exc + scission_matrix_ion

print('direct G(S) =', np.sum(scission_matrix) / np.sum(e_matrix_E_dep) * 100)
