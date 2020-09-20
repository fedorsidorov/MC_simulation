# %%
import importlib
import constants
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
from mapping import mapping_3p3um_80nm as mapping
from functions import MC_functions as mcf
from functions import DEBER_functions as deber
from functions import mapping_functions as mf
from functions import diffusion_functions as df
from functions import reflow_functions as rf
from functions import plot_functions as pf
from functions import evolver_functions as ef

from scipy.optimize import curve_fit

mapping = importlib.reload(mapping)
deber = importlib.reload(deber)
mcf = importlib.reload(mcf)
mf = importlib.reload(mf)
df = importlib.reload(df)
ef = importlib.reload(ef)
rf = importlib.reload(rf)
pf = importlib.reload(pf)

# %%
resist_matrix = np.load('data/exp_3p3um_80nm/resist_matrix.npy')
chain_lens = np.load('data/exp_3p3um_80nm/chain_lens.npy')
n_chains = len(chain_lens)

chain_tables = []
progress_bar = tqdm(total=n_chains, position=0)

for n in range(n_chains):
    chain_tables.append(
        np.load('data/exp_3p3um_80nm/chain_tables/chain_table_' + str(n) + '.npy'))
    progress_bar.update()

resist_shape = mapping.hist_5nm_shape

# %%
l_x = mapping.l_x * 1e-7
l_y = mapping.l_y * 1e-7
area = l_x * l_y
d_PMMA = mapping.z_max * 1e-7  # cm
j_exp_s = 1.9e-9  # A / cm
j_exp_l = 1.9e-9 * l_x
dose_s = 0.6e-6  # C / cm^2
dose_l = dose_s * l_x
t = dose_l / j_exp_l  # 316 s
dt = 1  # s
Q = dose_s * area
n_electrons = Q / constants.e_SI  # 2 472
n_electrons_s = int(np.around(n_electrons / t))

r_beam = 150e-7

zip_length = 1000
T_C = 125
Tg = 120
dT = T_C - Tg

time_s = 10

xx = mapping.x_centers_50nm * 1e-7  # cm
zz_vac = np.zeros(len(xx))  # cm
monomer_matrix = np.zeros(np.shape(resist_matrix)[:3])

zz_vac_list = [zz_vac]

for i in range(32):

    print('!!!!!!!!!', i, '!!!!!!!!!')

    plt.figure(dpi=300)
    plt.plot(xx, zz_vac)
    plt.title('suda herachat electrons, i = ' + str(i))
    plt.show()

    print('getting e_DATA ...')
    e_DATA, e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(
        xx=xx,
        zz_vac=zz_vac,
        d_PMMA=d_PMMA,
        n_electrons=n_electrons_s*time_s,
        E0=20e+3,
        r_beam=r_beam
    )
    print('e_DATA is obtained')

    print('getting scission_matrix ...')
    scission_matrix, E_dep_matrix = deber.get_scission_matrix(e_DATA, weight=0.35)
    print('scission_matrix is obtained')

    print('process mapping ...')
    mf.process_mapping(scission_matrix, resist_matrix, chain_tables)
    print('mapping is carried out')

    print('process depolymerization ...')
    mf.process_depolymerization(resist_matrix, chain_tables, zip_length)
    print('depolymerization is carried out')

    print('getting chain len matrix ...')
    sum_m, sum_m2, new_monomer_matrix = mf.get_chain_len_matrix(resist_matrix, chain_tables)
    print('chain len matrix is obtained')

    np.save('sum_m_' + str(i) + '.npy', sum_m)
    np.save('sum_m2_' + str(i) + '.npy', sum_m2)
    np.save('monomer_matrix' + str(i) + '.npy', monomer_matrix)

# %%
