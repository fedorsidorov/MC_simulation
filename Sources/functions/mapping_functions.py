import importlib
from collections import deque
import copy
import numpy as np
from tqdm import tqdm
import constants as const
import constants
import indexes

indexes = importlib.reload(indexes)
constants = importlib.reload(constants)


# %%
def move_scissions(scission_matrix, x_ind, y_ind, z_ind, n_sci):

    scission_matrix[x_ind, y_ind, z_ind] -= n_sci

    if x_ind + 1 < np.shape(scission_matrix)[0]:
        scission_matrix[x_ind + 1, y_ind, z_ind] += n_sci
    elif y_ind + 1 < np.shape(scission_matrix)[1]:
        scission_matrix[x_ind, y_ind + 1, z_ind] += n_sci
    elif z_ind + 1 < np.shape(scission_matrix)[2]:
        scission_matrix[x_ind, y_ind, z_ind + 1] += n_sci
    else:
        scission_matrix[x_ind, y_ind, z_ind] += n_sci
        print('no space for extra events, nowhere to move')


def rewrite_monomer_type(resist_matrix, chain_table, n_monomer, new_type):
    chain_table[n_monomer, indexes.monomer_type_ind] = new_type
    x_ind, y_ind, z_ind, monomer_line_pos = chain_table[n_monomer, :indexes.monomer_type_ind].astype(int)
    resist_matrix[x_ind, y_ind, z_ind, monomer_line_pos, indexes.monomer_type_ind] = new_type


# def process_scission(resist_matrix, chain_table, n_monomer, monomer_type):
def process_scission(resist_matrix, chain_table, n_monomer):
    monomer_type = chain_table[n_monomer, indexes.monomer_type_ind]

    if monomer_type == indexes.middle_monomer:  # bonded monomer
        # choose between left and right bond
        new_monomer_type = np.random.choice([0, 2])
        rewrite_monomer_type(resist_matrix, chain_table, n_monomer, new_monomer_type)
        n_next_monomer = n_monomer + new_monomer_type - 1
        next_x_ind, next_y_ind, next_z_ind, _, next_monomer_type = chain_table[n_next_monomer]

        # if next monomer is at the end
        if next_monomer_type in [indexes.begin_monomer, indexes.end_monomer]:
            next_monomer_new_type = indexes.free_monomer
            rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

        # if next monomer is fully bonded
        elif next_monomer_type == indexes.middle_monomer:
            next_monomer_new_type = next_monomer_type - (new_monomer_type - 1)
            rewrite_monomer_type(resist_matrix, chain_table, n_next_monomer, next_monomer_new_type)

        else:
            pass
            # print('next monomer type error, next_monomer_type =', monomer_type, next_monomer_type)

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
            pass
            # print('next monomer type error, next_monomer_type =', monomer_type, next_monomer_type)

    else:
        pass
        # print('monomer type error, monomer_type =', monomer_type)


def process_mapping(scission_matrix, resist_matrix, chain_tables):

    resist_shape = np.shape(scission_matrix)
    progress_bar = tqdm(total=resist_shape[0], position=0)

    for x_ind in range(resist_shape[0]):
        for y_ind in range(resist_shape[1]):
            for z_ind in range(resist_shape[2]):

                n_scissions = int(scission_matrix[x_ind, y_ind, z_ind])

                while n_scissions:

                    monomer_positions = list(
                        np.where(resist_matrix[x_ind, y_ind, z_ind, :, indexes.monomer_type_ind] <=
                                 indexes.end_monomer)[0]
                    )

                    if len(monomer_positions) == 0:  # move events to one of further bins
                        # print('moving')
                        # print(x_ind, y_ind, z_ind)
                        move_scissions(scission_matrix, x_ind, y_ind, z_ind, n_scissions)
                        break

                    monomer_pos = np.random.choice(monomer_positions)
                    n_scissions -= 1

                    n_chain, n_monomer, monomer_type = resist_matrix[x_ind, y_ind, z_ind, monomer_pos, :]
                    chain_table = chain_tables[n_chain]

                    # process_scission(resist_matrix, chain_table, n_monomer, monomer_type)
                    process_scission(resist_matrix, chain_table, n_monomer)

        progress_bar.update()

    # return resist_matrix, chain_tables


def process_depolymerization(resist_matrix, chain_tables, zip_length):

    for ct_num, ct in enumerate(chain_tables):

        sci_inds = np.where(ct[1:, indexes.monomer_type_ind] == 0)[0] + 1

        for sci_ind in sci_inds:
            now_table = ct
            n_mon = sci_ind
            step = np.random.choice([-1, 1])

            if step == -1:
                n_mon -= 1

            rewrite_monomer_type(resist_matrix, now_table, n_mon, indexes.free_monomer)
            kin_len = 1
            n_mon += step

            while kin_len < zip_length:  # do talogo

                x_bin, y_bin, z_bin, _, mon_type = now_table[n_mon, :]

                if mon_type <= indexes.end_monomer and (n_mon > 0) and (n_mon < (len(now_table) - 1)):
                    rewrite_monomer_type(resist_matrix, now_table, n_mon, indexes.free_monomer)
                    kin_len += 1
                    n_mon += step

                else:  # chain transfer
                    if mon_type != indexes.free_monomer:
                        rewrite_monomer_type(resist_matrix, now_table, n_mon, indexes.free_monomer)
                        kin_len += 1

                    # free_line_inds = np.where(resist_matrix[x_bin, y_bin, z_bin, :] <= indexes.end_monomer)[0]
                    free_line_inds = np.where(resist_matrix[x_bin, y_bin, z_bin, :, indexes.monomer_type_ind] <=
                                              indexes.end_monomer)[0]

                    # if len(free_line_inds) == 0:
                    #     print('no free indexes')
                    #     if y_bin + 1 < np.shape(resist_matrix)[1]:
                    #         y_bin += 1
                    #     elif z_bin + 1 < np.shape(resist_matrix)[2]:
                    #         z_bin += 1
                    #     elif x_bin + 1 < np.shape(resist_matrix)[0]:
                    #         x_bin += 1
                    #     break

                    while len(free_line_inds) == 0:
                        # print('no free indexes')
                        if y_bin + 1 < np.shape(resist_matrix)[1]:
                            y_bin += 1
                            free_line_inds = np.where(resist_matrix[x_bin, y_bin, z_bin, :, indexes.monomer_type_ind] <=
                                                      indexes.end_monomer)[0]
                        elif z_bin + 1 < np.shape(resist_matrix)[2]:
                            z_bin += 1
                            free_line_inds = np.where(resist_matrix[x_bin, y_bin, z_bin, :, indexes.monomer_type_ind] <=
                                                      indexes.end_monomer)[0]
                        elif x_bin + 1 < np.shape(resist_matrix)[0]:
                            x_bin += 1
                            free_line_inds = np.where(resist_matrix[x_bin, y_bin, z_bin, :, indexes.monomer_type_ind] <=
                                                      indexes.end_monomer)[0]
                        else:
                            break

                    if len(free_line_inds) == 0:
                        break

                    new_line_ind = np.random.choice(free_line_inds)
                    new_chain_num, new_n_mon = resist_matrix[x_bin, y_bin, z_bin, new_line_ind, :2]

                    now_table = chain_tables[new_chain_num]
                    n_mon = new_n_mon
                    step = np.random.choice([-1, 1])


def get_chain_len_matrix(resist_matrix, chain_tables):

    sum_m2 = np.zeros(np.shape(resist_matrix)[:3])  # for chains only!!!
    sum_m = np.zeros(np.shape(resist_matrix)[:3])  # for chains only!!!
    monomer_matrix = np.zeros(np.shape(resist_matrix)[:3])

    progress_bar = tqdm(total=len(chain_tables), position=0)

    for ct_num, ct in enumerate(chain_tables):

        now_len = 0
        bins = []

        for n_mon, mon_line in enumerate(ct):

            x_bin, y_bin, z_bin, _, mon_type = mon_line

            if mon_type != indexes.free_monomer:
                now_len += 1
                bins.append([x_bin, y_bin, z_bin])

            else:
                monomer_matrix[x_bin, y_bin, z_bin] += 1
                rewrite_monomer_type(resist_matrix, ct, n_mon, indexes.gone_monomer)

                if now_len > 0:  # chain length is gathered
                    for bu in np.unique(bins, axis=0):
                        ind_x, ind_y, ind_z = bu
                        now_mass = now_len * 100
                        sum_m2[ind_x, ind_y, ind_z] += now_mass ** 2
                        sum_m[ind_x, ind_y, ind_z] += now_mass

                    now_len = 0
                    bins = []

        if now_len > 0:
            now_mass = now_len * 100

            for bu in np.unique(bins, axis=0):
                ind_x, ind_y, ind_z = bu
                sum_m2[ind_x, ind_y, ind_z] += now_mass ** 2
                sum_m[ind_x, ind_y, ind_z] += now_mass

        progress_bar.update()

    # monomer_matrix_2d = np.sum(monomer_matrix, axis=1)

    return sum_m, sum_m2, monomer_matrix


def get_local_Mw_matrix(sum_m_1d, sum_m2_1d, monomer_matrix_1d):

    # sum_m_1d = np.sum(np.sum(sum_m, axis=1), axis=1)
    # sum_m2_1d = np.sum(np.sum(sum_m2, axis=1), axis=1)
    # monomer_matrix_1d = np.sum(np.sum(monomer_matrix, axis=1), axis=1)

    # matrix_Mw = np.zeros(np.shape(sum_m))
    matrix_Mw_1d = np.zeros(np.shape(sum_m_1d))

    # for i in range(np.shape(sum_m)[0]):
    #     for j in range(np.shape(sum_m)[1]):
    #         for k in range(np.shape(sum_m)[2]):
    #             matrix_Mw[i, j, k] = (sum_m2[i, j, k] + (1 * 100) ** 2 * monomer_matrix[i, j, k]) / \
    #                                  (sum_m[i, j, k] + (1 * 100) * monomer_matrix[i, j, k])

    for i in range(len(sum_m_1d)):
        matrix_Mw_1d[i] = (sum_m2_1d[i] + (1 * 100) ** 2 * monomer_matrix_1d[i]) / \
            (sum_m_1d[i] + (1 * 100) * monomer_matrix_1d[i])

    # return matrix_Mw
    return matrix_Mw_1d

