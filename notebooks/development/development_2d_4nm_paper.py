import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import mapping_aktary as mapping
from functions import development_functions_2d as df

mapping = importlib.reload(mapping)
df = importlib.reload(df)

# %% 2nm histograms
sum_lens_matrix = np.load('data/chains/combine_chains/development/sum_lens_matrix_series_1_1nm_1500.npy')
n_chains_matrix = np.load('data/chains/combine_chains/development/n_chains_matrix_series_1_1nm_1500.npy')

n_chains_matrix_avg = np.average(n_chains_matrix, axis=1)
sum_lens_matrix_avg = np.average(sum_lens_matrix, axis=1)

local_chain_length_avg = sum_lens_matrix_avg / n_chains_matrix_avg

font_size = 8
# font_size = 14

_, ax = plt.subplots(dpi=300)
# fig = plt.figure(figsize=[3.35, 3], dpi=300)
fig = plt.gcf()
fig.set_size_inches(3, 3)

plt.imshow(local_chain_length_avg.transpose())
# plt.imshow(np.log10(local_chain_length_avg).transpose())

cb = plt.colorbar()
cb.ax.tick_params(labelsize=font_size)

ax.set_xlabel('$x$, nm', fontsize=font_size)
ax.set_ylabel('$y$, nm', fontsize=font_size)

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.show()

# %
# plt.savefig('hist2.eps', bbox_inches='tight')
# plt.savefig('hist2_new.tiff', bbox_inches='tight')
