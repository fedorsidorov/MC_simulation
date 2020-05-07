import importlib

import numpy as np
from tqdm import tqdm

# import mapping_harris as mapping
import indexes
import mapping_aktary as mapping

mapping = importlib.reload(mapping)
indexes = importlib.reload(indexes)


# %%
def move_scissions(scission_matrix, x_ind, y_ind, z_ind, n_sci):
    if x_ind + 1 < np.shape(scission_matrix)[0]:
        scission_matrix[x_ind + 1, y_ind, z_ind] += n_sci
    elif y_ind + 1 < np.shape(scission_matrix)[1]:
        scission_matrix[x_ind, y_ind + 1, z_ind] += n_sci
    elif z_ind + 1 < np.shape(scission_matrix)[2]:
        scission_matrix[x_ind, y_ind, z_ind + 1] += n_sci
    else:
        print('no space for extra events, nowhere to move')


def rewrite_monomer_type(resist_matrix, chain_table, n_monomer, new_type):
    chain_table[n_monomer, indexes.monomer_type_ind] = new_type
    x_ind, y_ind, z_ind, monomer_line_pos = chain_table[n_monomer, :indexes.monomer_type_ind].astype(int)
    resist_matrix[x_ind, y_ind, z_ind, monomer_line_pos, indexes.monomer_type_ind] = new_type


def process_scission(resist_matrix, chain_table, n_monomer, monomer_type):
    if monomer_type == indexes.middle_monomer:  # bonded monomer
        # choose between left and right bond
        new_monomer_type = np.random.choice([0, 2])
        rewrite_monomer_type(resist_matrix, chain_table, n_monomer, new_monomer_type)
        n_next_monomer = n_monomer + new_monomer_type - 1
        next_x_ind, next_y_ind, next_z_ind, _, next_monomer_type = chain_table[n_next_monomer]

        # if next monomer was at the end
        if next_monomer_type in [indexes.begin_monomer, indexes.end_monomer]:
            next_monomer_new_type = indexes.free_monomer
            rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

        # if next monomer is full bonded
        elif next_monomer_type == indexes.middle_monomer:
            next_monomer_new_type = next_monomer_type - (new_monomer_type - 1)
            rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

        else:
            print('next monomer type error, next_monomer_type =', monomer_type, next_monomer_type)

    elif monomer_type in [indexes.begin_monomer, indexes.end_monomer]:  # half-bonded monomer
        new_monomer_type = indexes.free_monomer
        rewrite_monomer_type(resist_matrix, chain_table, n_monomer, new_monomer_type)
        n_next_monomer = n_monomer - (monomer_type - 1)  # minus, Karl!
        next_x_ind, next_y_ind, next_z_ind, _, next_monomer_type = chain_table[n_next_monomer]

        # if next monomer was at the end
        if next_monomer_type in [indexes.begin_monomer, indexes.end_monomer]:
            next_monomer_new_type = indexes.free_monomer
            rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

        # if next monomer is full bonded
        elif next_monomer_type == indexes.middle_monomer:
            next_monomer_new_type = next_monomer_type + (monomer_type - 1)
            rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

        else:
            print('next monomer type error, next_monomer_type =', monomer_type, next_monomer_type)

    else:
        print('monomer type error, monomer_type =', monomer_type)


def get_chain_lens(chain_tables):
    lens_final = []
    p_bar = tqdm(total=len(chain_tables), position=0)

    for chain_table in chain_tables:
        cnt = 0
        if len(chain_table) == 1:
            lens_final.append(1)
            continue
        for line in chain_table:
            monomer_type = line[indexes.monomer_type_ind]
            if monomer_type == 0:
                cnt = 1
            elif monomer_type == 1:
                cnt += 1
            elif monomer_type == 2:
                cnt += 1
                lens_final.append(cnt)
                cnt = 0
        p_bar.update()

    return np.array(lens_final)


def get_local_chain_len(resist_shape, N_mon_max, chain_table):
    chain_sum_len_matrix = np.zeros(resist_shape)
    n_chains_matrix = np.zeros(resist_shape)

    for idx, chain in enumerate(chain_table):
        beg_pos = 0

        while True:

            if beg_pos >= N_mon_max or chain[beg_pos, mapping.monomer_type_pos] == mapping.uint16_max:
                break

            if chain[beg_pos, mapping.monomer_type_pos] in [mapping.free_monomer, mapping.free_radicalized_monomer]:
                beg_pos += 1
                continue

            if chain[beg_pos, mapping.monomer_type_pos] != mapping.begin_monomer:
                print('monomer_type', chain[beg_pos, mapping.monomer_type_pos])
                print('idx, beg_pos', idx, beg_pos)
                print('chain index_indng error!')

            where_result = np.where(chain[beg_pos:, mapping.monomer_type_pos] == mapping.end_monomer)[0]

            if len(where_result) == 0:
                break

            end_pos = beg_pos + where_result[0]
            now_chain_len = end_pos - beg_pos

            inds_list = []

            for mon_line in chain[beg_pos:end_pos + 1]:

                x_pos, y_pos, z_pos = mon_line[:3]

                if x_pos == y_pos == z_pos == mapping.uint16_max:
                    continue

                now_poss = [x_pos, y_pos, z_pos]

                if now_poss in inds_list:
                    continue

                chain_sum_len_matrix[x_pos, y_pos, z_pos] += now_chain_len
                n_chains_matrix[x_pos, y_pos, z_pos] += 1

                inds_list.append(now_poss)

            beg_pos = end_pos + 1

    return chain_sum_len_matrix, n_chains_matrix
