import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import constants as cp
from mapping import mapping_viscosity_80nm as mapping

mapping = importlib.reload(mapping)
cp = importlib.reload(cp)

# %%
bin_size = '10nm'

chain_lens = np.load('/Volumes/ELEMENTS/chains_viscosity_80nm/' + bin_size + '/prepared_chains_1/chain_lens.npy')
hist_10nm = np.load('/Volumes/ELEMENTS/chains_viscosity_80nm/' + bin_size + '/rot_sh_sn_chains_1/hist_10nm.npy')

n_monomers = np.sum(chain_lens)
