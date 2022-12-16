import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %% 150C 100s
xx_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/xx_366_zero.npy')
zz_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/zz_366_zero.npy')

path = '/Volumes/Transcend/SIM_DEBER/150C_100s_final/'
xx_bins = np.load(path + 'try_0/xx_bins.npy')

n_tries = 100

zz_bins_sum = np.zeros(len(xx_bins))
