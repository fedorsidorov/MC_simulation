import numpy as np
import importlib
from tqdm import tqdm
from mapping import mapping_3p3um_80nm as mapping
from functions import reflow_functions as rf

mapping = importlib.reload(mapping)
rf = importlib.reload(rf)

# %%
scission_array = np.load('scission_array_sum.npy')
n_scissions_arr_100nm = np.sum(scission_array[:20])
n_scissions_total = np.sum(n_scissions_arr_100nm) * 330

scission_matrix = np.zeros(mapping.hist_5nm_shape)

for i in range(int(n_scissions_total)):

    x_ind = np.random.choice(len(mapping.x_centers_5nm))
    y_ind = np.random.choice(len(mapping.y_centers_5nm))
    z_ind = np.random.choice(len(mapping.z_centers_5nm))

    scission_matrix[x_ind, y_ind, z_ind] += 1


# %%
n_scissions_500nm = np.sum(scission_array[:100])
n_scissions = np.sum(n_scissions_500nm)

chain_lens_list = []

chain_lens = np.load('/Volumes/ELEMENTS/chains_viscosity/prepared_chains_1/chain_lens.npy')[:1337]  # 500 nm

for chain_len in chain_lens:
    chain_lens_list.append(chain_len)


progress_bar = tqdm(total=int(n_scissions), position=0)

# for i in range(int(n_scissions)):
#     chain_len = np.random.choice(chain_lens_list)
#
#     while chain_len < 0:
#         chain_len = np.random.choice(chain_lens_list)
#
#     chain_lens_list.remove(chain_len)
#
#     pos = np.random.choice(int(chain_len))
#     pos_0 = pos
#
#     step = np.random.choice((-1, 1))
#
#     n_steps = 0
#
#     for j in range(2000):
#         if pos + step == -1:
#             # chain_lens_list.append(value)
#             break
#         elif pos + step == chain_len:
#             # chain_lens_list.append(pos_0)
#             break
#
#         pos += step
#         n_steps += 1
#
#     if pos < 0:
#         print('negative!')
#
#     if n_steps == 2000:
#         if step == -1:
#             chain_lens_list.append(pos)
#         elif step == 1:
#             if chain_len - pos < 0:
#                 print('!!!!!!!!!!!!!', step, pos_0, pos)
#             chain_lens_list.append(chain_len - pos)

# %%
chain_lens_array = np.array(chain_lens_list)

# %%
print('Mw =', np.sum(chain_lens_array ** 2) / np.sum(chain_lens_array) * 100)




# %%
Mw = np.sum(np.array(chain_mass_list) ** 2) / np.sum(np.array(chain_mass_list))

print('Mw =', Mw)
print('log viscosity =', np.log10(rf.get_viscosity_W(120, Mw)))

# %%
total_portions = int(1e+3 * 1e+5)

now_portion = [100] * int(1e+6)

chain_mass_list += now_portion

Mw = np.sum(np.array(chain_mass_list) ** 2) / np.sum(np.array(chain_mass_list))
print(np.log10(rf.get_viscosity_W(120, Mw)))
