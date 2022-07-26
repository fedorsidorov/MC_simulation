import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import constants as const
import grid
from functions import MC_functions as mcf

const = importlib.reload(const)
mcf = importlib.reload(mcf)


#%%
def get_elsepa_theta_diff_cs(filepath):

    # elsepa_file = np.loadtxt(filepath, skiprows=31)
    elsepa_file = np.loadtxt(filepath)
    return elsepa_file[:, 0], elsepa_file[:, 2]


def get_elsepa_EE_cs(dirname):

    # tcs_table = np.loadtxt(dirname + '/tcstable.dat', skiprows=7)
    tcs_table = np.loadtxt(dirname + '/tcstable.dat')
    return tcs_table[:, 0], tcs_table[:, 1]


# %%
EE = get_elsepa_EE_cs('notebooks/elastic/raw_data/atomic/C')[0]
theta = get_elsepa_theta_diff_cs('notebooks/elastic/raw_data/atomic/C/dcs_1p000e01.dat')[0]

# EE = np.array([
#           10,    11,    12,    13,    14,    15,    16,    17,    18,    19,  #  0- 9
#           20,    25,    30,    35,    40,    45,    50,    60,    70,    80,  # 10-19
#           90,   100,   150,   200,   250,   300,   350,   400,   450,   500,  # 20-29
#          600,   700,   800,   900,  1000,  1500,  2000,  2500,  3000,  3500,  # 30-39
#         4000,  4500,  5000,  6000,  7000,  8000,  9000, 10000, 11000, 12000,  # 40-49
#        13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000,  # 50-59
#        23000, 24000, 25000])  # 60-62


# %%
for model in ['easy', 'atomic', 'muffin']:
    for element in ['H', 'C', 'O', 'Si']:

        folder = os.path.join('notebooks/elastic/raw_data', model, element)

        diff_cs = np.zeros((len(EE), len(theta)))

        for i, E in enumerate(EE):

            E_str = str(int(E))

            d1 = E_str[0]
            d2 = E_str[1]
            exp = str(len(E_str) - 1)

            fname = 'dcs_' + d1 + 'p' + d2 + '00e0' + exp + '.dat'

            diff_cs[i, :] = get_elsepa_theta_diff_cs(os.path.join(folder, fname))[1]

        cs = get_elsepa_EE_cs(folder)[1]

        np.save(os.path.join('notebooks/elastic/raw_arrays', model, element, element + '_' + model +
                             '_diff_cs.npy'), diff_cs)
        np.save(os.path.join('notebooks/elastic/raw_arrays', model, element, element + '_' + model +
                             '_cs.npy'), cs)

# %%
MELEC = '4'
MEXCH = '1'
# MCPOL = '0'

# for MELEC in ['1', '2', '3', '4']:
# for MEXCH in ['0', '1', '2', '3']:
for MCPOL in ['0', '1', '2']:

    folder = 'notebooks/elastic/raw_data/root_Hg/root_' + MELEC + MEXCH + MCPOL
    # folder = 'notebooks/elastic/raw_data/root_Si/root_' + MELEC + MEXCH + MCPOL

    diff_cs = np.zeros((len(EE), len(theta)))

    for i, E in enumerate(EE):

        E_str = str(int(E))

        d1 = E_str[0]
        d2 = E_str[1]
        exp = str(len(E_str) - 1)

        fname = 'dcs_' + d1 + 'p' + d2 + '00e0' + exp + '.dat'

        diff_cs[i, :] = get_elsepa_theta_diff_cs(os.path.join(folder, fname))[1]

    cs = get_elsepa_EE_cs(folder)[1]

    np.save('notebooks/elastic/raw_arrays/root_Hg/root_' + MELEC + MEXCH + MCPOL + '_diff_cs.npy', diff_cs)
    np.save('notebooks/elastic/raw_arrays/root_Hg/root_' + MELEC + MEXCH + MCPOL + '_cs.npy', cs)

    # np.save('notebooks/elastic/raw_arrays/root_Si/root_' + MELEC + MEXCH + MCPOL + '_diff_cs.npy', diff_cs)
    # np.save('notebooks/elastic/raw_arrays/root_Si/root_' + MELEC + MEXCH + MCPOL + '_cs.npy', cs)


