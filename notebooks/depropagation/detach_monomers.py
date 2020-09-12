import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mapping_exp_80_3 as mapping
from functions import mapping_functions as mf
import indexes
import copy

mapping = importlib.reload(mapping)
indexes = importlib.reload(indexes)
mf = importlib.reload(mf)

# %%
resist_matrix = np.load('data/exp_80_3/resist_matrix_5nm_80_3_exposed.npy')

chain_lens = np.load('data/exp_80_3/chain_lens_80_3.npy')
n_chains = len(chain_lens)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('data/exp_80_3/chain_tables_exposed_5nm/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = mapping.hist_5nm_shape

# %% rewrite monomer types
zip_len = 2000
no_way_num = 0

progress_bar = tqdm(total=n_chains, position=0)

for ct_num, ct in enumerate(chain_tables):

    sci_inds = np.where(ct[1:, 4] == 0)[0]

    for sci_ind in sci_inds:
        now_table = ct
        n_mon = sci_ind
        step = np.random.choice([-1, 1])

        if step == -1:
            n_mon -= 1

        mf.rewrite_monomer_type(resist_matrix, now_table, n_mon, indexes.free_monomer)
        kin_len = 1
        n_mon += step

        while kin_len < zip_len:  # do talogo

            x_bin, y_bin, z_bin, mon_line_pos, mon_type = now_table[n_mon, :]

            if mon_type == 1:  # chain is not ended
                mf.rewrite_monomer_type(resist_matrix, now_table, n_mon, indexes.free_monomer)
                kin_len += 1
                n_mon += step

            else:  # chain transfer
                # print('chain_transfer')
                mf.rewrite_monomer_type(resist_matrix, now_table, n_mon, indexes.free_monomer)
                kin_len += 1

                free_line_inds = np.where(resist_matrix[x_bin, y_bin, z_bin, :] == 1)[0]

                if len(free_line_inds) == 0:
                    # print('no way')
                    no_way_num += 1
                    break

                new_line_ind = np.random.choice(free_line_inds)
                new_chain_num, new_n_mon = resist_matrix[x_bin, y_bin, z_bin, new_line_ind, :2]

                now_table = chain_tables[new_chain_num]
                n_mon = new_n_mon

    progress_bar.update()

# %%
sum_lens = np.zeros(np.shape(resist_matrix)[:3])
n_chains = np.zeros(np.shape(resist_matrix)[:3])

progress_bar = tqdm(total=len(chain_lens), position=0)

for ct in chain_tables:

    now_len = 0
    bins = []

    for i, line in enumerate(ct):

        x_bin, y_bin, z_bin, mon_line_pos, mon_type = line
        bins.append([x_bin, y_bin, z_bin])

        if mon_type == indexes.free_monomer:

            if now_len > 0:
                now_x, now_y, now_z = bins[0]
                sum_lens[now_x, now_y, now_z] += now_len
                n_chains[now_x, now_y, now_z] += 1

                for xyz in bins[1:]:
                    new_x, new_y, new_z = xyz
                    if not (new_x == now_x and new_y == now_y and new_z == now_z):
                        sum_lens[new_x, new_y, new_z] += now_len
                        n_chains[new_x, new_y, new_z] += 1
                        now_x = new_x
                        now_y = new_y
                        now_z = new_z

                now_len = 0

            sum_lens[x_bin, y_bin, z_bin] += 1
            n_chains[x_bin, y_bin, z_bin] += 1

        else:
            now_len += 1

    progress_bar.update()

# %%
sum_lens_2d = np.sum(sum_lens, axis=1)
n_chains_2d = np.sum(n_chains, axis=1)

sum_lens_1d = np.sum(sum_lens_2d, axis=1)
n_chains_1d = np.sum(n_chains_2d, axis=1)

ans = sum_lens_1d / n_chains_1d

plt.figure(dpi=300)
plt.plot(mapping.x_centers_5nm, ans)

plt.xlabel('x, nm')
plt.ylabel('average chain length')
# plt.ylim(0, 1500)
plt.grid()
# plt.show()
plt.savefig('average_len.png')
