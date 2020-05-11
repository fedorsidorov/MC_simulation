import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import chain_functions as cf
from functions import array_functions as af
from functions import plot_functions as pf
from tqdm import tqdm
# import mapping_harris as mapping
import mapping_aktary as mapping
import constants as const

cf = importlib.reload(cf)
af = importlib.reload(af)
pf = importlib.reload(pf)
mapping = importlib.reload(mapping)
const = importlib.reload(const)

# %%
folder_name = 'Aktary'
progress_bar = tqdm(total=754, position=0)

hist_4nm = np.zeros(mapping.hist_4nm_shape)

for chain_num in range(754):
    now_chain = np.load('data/chains/' + folder_name +
                        '/best_sh_sn_chains/sh_sn_chain_' + str(chain_num) + '.npy')
    hist_4nm += np.histogramdd(now_chain, bins=mapping.bins_4nm)[0]
    progress_bar.update()

progress_bar.close()

# %%
plt.imshow(np.average(hist_4nm, axis=0))
plt.show()

# %% save chains to files
# data/chains/Aktary/shifted_snaked_chains
dest_folder = 'data/chains/' + folder_name + '/shifted_snaked_chains/'
progress_bar = tqdm(total=len(chain_list), position=0)

for n, chain in enumerate(final_chain_list):
    np.save(dest_folder + 'shifted_snaked_chain_' + str(n) + '.npy', chain)
    progress_bar.update(1)

np.save(dest_folder + 'hist_2nm.npy', hist_2nm)
