import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

import constants as const
from functions import MC_functions as mcf

const = importlib.reload(const)
mcf = importlib.reload(mcf)


#%%
for kind in ['easy', 'atomic', 'muffin']:

    cs_H = np.load('final_arrays/' + kind + '/H/H_' + kind + '_cs_extrap.npy')
    cs_C = np.load('final_arrays/' + kind + '/C/C_' + kind + '_cs_extrap.npy')
    cs_O = np.load('final_arrays/' + kind + '/O/O_' + kind + '_cs_extrap.npy')
    cs_Si = np.load('final_arrays/' + kind + '/simple_Si_MC/Si_' + kind + '_cs_extrap.npy')

    cs_MMA = const.N_H_MMA * cs_H + const.N_C_MMA * cs_C + const.N_O_MMA * cs_O
    
    u_PMMA = cs_MMA * const.n_MMA
    u_Si = cs_Si * const.n_Si

    # plt.loglog(mc.EE, cs_MMA)
    
    np.save('final_arrays/PMMA/MMA_' + kind + '_cs.npy', cs_MMA)
    np.save('final_arrays/PMMA/PMMA_' + kind + '_u.npy', u_PMMA)
    
    np.save('final_arrays/simple_Si_MC/Si_' + kind + '_cs.npy', cs_Si)
    np.save('final_arrays/simple_Si_MC/Si_' + kind + '_u.npy', u_Si)
