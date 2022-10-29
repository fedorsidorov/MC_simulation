import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# import indexes as ind
from functions import array_functions as af
from functions import e_matrix_functions as emf
# from functions import plot_functions as pf
# from functions import scission_functions as sf
# from functions import G_functions as Gf

from mapping import mapping_harris as mapping

mapping = importlib.reload(mapping)
emf = importlib.reload(emf)
# ind = importlib.reload(ind)
af = importlib.reload(af)
# pf = importlib.reload(pf)
# sf = importlib.reload(sf)
# Gf = importlib.reload(Gf)

# %%
e_matrix_val = np.zeros(mapping.hist_5nm_shape)
e_matrix_E_dep = np.zeros(mapping.hist_5nm_shape)

n_electrons = 0

source = '/Users/fedor/PycharmProjects/MC_simulation/data/4Harris/'

primary_electrons_in_file = 100
# n_files = 700
file_cnt = 0

# progress_bar = tqdm(total=n_electrons_required, position=0)

plt.figure(dpi=300)

# while n_electrons < n_electrons_required:
while file_cnt < 1:

    # print(file_cnt)
    now_e_DATA = np.load(source + 'e_DATA_Pn_' + str(file_cnt) + '.npy')
    file_cnt += 1

    # check PMMA and inelastic events
    now_e_DATA = now_e_DATA[
        np.where(
            np.logical_and(now_e_DATA[:, 2] == 0, now_e_DATA[:, 3] == 1)
        )
    ]

    plt.plot(now_e_DATA[:, 4], now_e_DATA[:, 6], '.')

plt.xlim(-50, 50)
plt.ylim(0, 100)
plt.show()
