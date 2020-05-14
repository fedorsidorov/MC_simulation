import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import chain_functions as cf
from functions import array_functions as af
from functions import plot_functions as pf
from tqdm import tqdm
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

hist_1nm = np.zeros(mapping.hist_1nm_shape)

for chain_num in range(754):
    now_chain = np.load('data/chains/Aktary/best_sh_sn_chains/sh_sn_chain_' + str(chain_num) + '.npy')
    hist_1nm += np.histogramdd(now_chain, bins=mapping.bins_1nm)[0]
    progress_bar.update()

progress_bar.close()

# %%
plt.imshow(np.average(hist_1nm, axis=0))
plt.show()

# %% save hist_1nm
np.save('data/chains/Aktary/best_sh_sn_chains/best_hist_1nm.npy', hist_1nm)
