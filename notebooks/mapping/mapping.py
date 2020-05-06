import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
# import copy
from tqdm import tqdm

import constants_mapping as const_m

const_m = importlib.reload(const_m)


# %%
def move_scissions(sci_mat, xi, yi, zi, n_sci):
    if xi + 1 < np.shape(sci_mat)[0]:
        sci_mat[xi + 1, yi, zi] += n_sci
    elif yi + 1 < np.shape(sci_mat)[1]:
        sci_mat[xi, yi + 1, zi] += n_sci
    elif zi + 1 < np.shape(sci_mat)[2]:
        sci_mat[xi, yi, zi + 1] += n_sci
    else:
        print('no space for extra events, nowhere to move')


def rewrite_monomer_type(res_mat, ch_tab, n_mon, new_type):
    ch_tab[n_mon, const_m.monomer_type_ind] = new_type
    xi, yi, zi, mon_line_pos = ch_tab[n_mon, :const_m.monomer_type_ind].astype(int)
    res_mat[xi, yi, zi, mon_line_pos, const_m.monomer_type_ind] = new_type


# %%
e_matrix_val_exc_sci = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/e_matrix/C-C2:4_C-C\':2/e_matrix_val_exc_sci.npy')
e_matrix_val_ion_sci = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/e_matrix/C-C2:4_C-C\':2/e_matrix_val_ion_sci.npy')
scission_matrix = e_matrix_val_exc_sci + e_matrix_val_exc_sci

resist_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/data/Harris/MATRIX_resist_1.npy')
chain_lens_array = np.load(
    '/Users/fedor/PycharmProjects/MC_simulation/data/Harris/prepared_chains_1/prepared_chain_lens.npy')
lens_before = chain_lens_array
n_chains = len(chain_lens_array)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('/Users/fedor/PycharmProjects/MC_simulation/data/Harris/chain_tables_1/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = const_m.hist_2nm_shape

# %%
n_scissions_moved = 0
progress_bar = tqdm(total=resist_shape[0], position=0)

for x_ind in range(resist_shape[0]):
    for y_ind in range(resist_shape[1]):
        for z_ind in range(resist_shape[2]):

            if y_ind == z_ind == 0:
                progress_bar.update()

            n_scissions = int(scission_matrix[x_ind, y_ind, z_ind])
            monomer_positions = \
                list(np.where(resist_matrix[x_ind, y_ind, z_ind, :, const_m.n_chain_ind] != const_m.uint32_max)[0])

            # for _ in range(n_scissions):
            while n_scissions:

                if len(monomer_positions) == 0:  # move events to one of further bins
                    move_scissions(scission_matrix, x_ind, y_ind, z_ind, n_scissions)
                    n_scissions_moved += 1
                    break

                monomer_pos = np.random.choice(monomer_positions)
                monomer_positions.remove(monomer_pos)
                n_scissions -= 1

                n_chain, n_monomer, monomer_type = resist_matrix[x_ind, y_ind, z_ind, monomer_pos, :]
                chain_table = chain_tables[n_chain]

                if monomer_type != chain_table[n_monomer, -1]:
                    print('FUKKK!!', n_chain, n_monomer)
                    print(monomer_type, chain_table[n_monomer, -1])

                # if len(chain_table) == 1:
                #     continue

                if monomer_type == const_m.middle_monomer:  # bonded monomer
                    # choose between left and right bond
                    new_monomer_type = np.random.choice([0, 2])
                    rewrite_monomer_type(resist_matrix, chain_table, n_monomer, new_monomer_type)
                    n_next_monomer = n_monomer + new_monomer_type - 1
                    next_xi, next_yi, next_zi, _, next_monomer_type = chain_table[n_next_monomer]

                    # if next monomer was at the end
                    if next_monomer_type in [const_m.begin_monomer, const_m.end_monomer]:
                        next_monomer_new_type = const_m.free_monomer
                        rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

                    # if next monomer is full bonded
                    elif next_monomer_type == const_m.middle_monomer:
                        next_monomer_new_type = next_monomer_type - (new_monomer_type - 1)
                        rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

                    else:
                        print('\nerror!')
                        print('n_chain', n_chain)
                        print('n_mon', n_monomer)
                        print('next_monomer_type', next_monomer_type)

                elif monomer_type in [const_m.begin_monomer, const_m.end_monomer]:  # half-bonded monomer
                    new_monomer_type = const_m.free_monomer
                    rewrite_monomer_type(resist_matrix, chain_table, n_monomer, new_monomer_type)
                    n_next_monomer = n_monomer - (monomer_type - 1)  # minus, Karl!
                    next_xi, next_yi, next_zi, _, next_monomer_type = chain_table[n_next_monomer]

                    # if next monomer was at the end
                    if next_monomer_type in [const_m.begin_monomer, const_m.end_monomer]:
                        next_monomer_new_type = const_m.free_monomer
                        rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

                    # if next monomer is full bonded
                    elif next_monomer_type == const_m.middle_monomer:
                        next_monomer_new_type = next_monomer_type + (monomer_type - 1)
                        rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

                    else:
                        print('error 2', next_monomer_type)

                else:
                    # print(monomer_type)
                    continue

# %%
lens_final = []

for i, now_chain in enumerate(chain_tables):

    mu.pbar(i, len(chain_tables))
    cnt = 0

    if len(now_chain) == 1:
        lens_final.append(cnt + 1)
        continue

    for line in now_chain:
        mon_type = line[mon_type_ind]

        if mon_type == 0:
            cnt == 1

        elif mon_type == 1:
            cnt += 1

        elif mon_type == 2:
            cnt += 1
            lens_final.append(cnt)
            cnt = 0

chain_lens_final = np.array(lens_final)

# %%
np.save('lens_final_' + deg_path + '.npy', chain_lens_final)

# %%
# deg_path = '2CC'
# deg_path = '2CC+05ester'
# deg_path = '2CC+ester'
# deg_path = 'CC+ester'
# deg_path = '2CC+ester+3CH'

chain_lens_final = np.load('lens_final_' + deg_path + '.npy')

Mn0 = np.average(np.load(os.path.join(mc.sim_folder,
                                      'PMMA_sim_Harris', 'Harris_chain_lens_2020.npy'
                                      )) * mc.M0)

Mn = np.average(chain_lens_final * mc.M0)

total_E_loss = np.sum(np.load(os.path.join(mc.sim_folder,
                                           'e-matrix_Harris', 'Harris_e_matrix_dE_' + deg_path + '.npy'
                                           )))

# E_loss_1e = total_E_loss / emf.get_n_electrons_2D(1e-4, 100, 100, 0)
ps = (1 / Mn - 1 / Mn0) * mc.M0  ## probability of chain scission
# Q = 1e-4 ## exposure dose per cm^2
d = mc.rho_PMMA * (500e-7)  ## sheet density, g/cm^2

# Gs = ( ps * d * mc.Na * (1/M0) ) / ( Q/mc.e * E_loss_1e ) * 100
Gs = (ps * d * mc.Na * (1 / mc.M0)) / (total_E_loss / (100e-7) ** 2) * 100

print('Gs =', Gs)

chain_lens_initial = np.load(os.path.join(mc.sim_folder,
                                          'PMMA_sim_Harris', 'Harris_chain_lens_2020.npy'
                                          ))

chain_lens_final = np.load('lens_final_' + deg_path + '.npy')

# distr_i = np.load(os.path.join(mc.sim_folder,
#        'PMMA_sim_Harris', 'harris_initial_distr.npy'
#        ))

distr_f = np.load(os.path.join(mc.sim_folder,
                               'PMMA_sim_Harris', 'harris_final_distr.npy'
                               ))

mass = np.array(chain_lens_final) * mc.u_PMMA

bins = np.logspace(2, 7.1, 21)

_, ax = plt.subplots()

chain_lens_hist = np.histogram(mass, bins=bins)

plt.hist(mass, bins, label='simulation')
plt.plot(distr_f[:, 0], distr_f[:, 1] * chain_lens_hist[0].max(), label='experiment')

plt.gca().set_xscale('log')
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

plt.title(deg_path + ', G = ' + str(np.round(Gs * 100) / 100))
plt.xlabel('molecular weight')
plt.ylabel('N$_{entries}$')

plt.xlim(1e+2, 1e+6)

plt.legend()
plt.grid()
plt.show()

plt.savefig('Harris_final_' + deg_path + '.png', dpi=300)
