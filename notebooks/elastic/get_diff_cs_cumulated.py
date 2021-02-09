#%% Import
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

    diff_cs_H = np.load(os.path.join('final_arrays', kind,  'H',  'H' + '_' + kind + '_diff_cs.npy'))
    diff_cs_C = np.load(os.path.join('final_arrays', kind,  'C',  'C' + '_' + kind + '_diff_cs.npy'))
    diff_cs_O = np.load(os.path.join('final_arrays', kind,  'O',  'O' + '_' + kind + '_diff_cs.npy'))
    
    diff_cs_Si = np.load(os.path.join('final_arrays', kind, 'Si', 'Si' + '_' + kind + '_diff_cs.npy'))
    diff_cs_MMA = const.N_H_MMA * diff_cs_H + const.N_C_MMA * diff_cs_C + const.N_O_MMA * diff_cs_O
    
    diff_cs_MMA_plane = np.zeros(np.shape(diff_cs_MMA))
    diff_cs_Si_plane = np.zeros(np.shape(diff_cs_Si ))
    
    diff_cs_MMA_plane_norm = np.zeros(np.shape(diff_cs_MMA))
    diff_cs_Si_plane_norm = np.zeros(np.shape(diff_cs_Si ))
    
    diff_cs_MMA_plane_norm_cumulated = np.zeros(np.shape(diff_cs_MMA))
    diff_cs_Si_plane_norm_cumulated = np.zeros(np.shape(diff_cs_Si ))

    for i in range(len(const.EE)):
        
        now_diff_cs_MMA_plane = diff_cs_MMA[i, :] * 2*np.pi * np.sin(const.THETA_rad)
        now_diff_cs_Si_plane = diff_cs_Si[i, :] * 2*np.pi * np.sin(const.THETA_rad)
        
        diff_cs_MMA_plane[i, :] = now_diff_cs_MMA_plane
        diff_cs_Si_plane[i, :] = now_diff_cs_Si_plane
        
        diff_cs_MMA_plane_norm[i, :] = now_diff_cs_MMA_plane / np.sum(now_diff_cs_MMA_plane)
        diff_cs_Si_plane_norm[i, :] = now_diff_cs_Si_plane / np.sum(now_diff_cs_Si_plane)
        
        diff_cs_MMA_plane_norm_cumulated[i, :] = mcf.get_cumulated_array(diff_cs_MMA_plane_norm[i, :])
        diff_cs_Si_plane_norm_cumulated[i, :]  = mcf.get_cumulated_array(diff_cs_Si_plane_norm[i, :])
    
    
    np.save('final_arrays/PMMA/diff_cs_plane_norm_' + kind + '.npy', diff_cs_MMA_plane_norm)
    np.save('final_arrays/PMMA/diff_cs_plane_norm_cumulated_' + kind + '.npy',
            diff_cs_MMA_plane_norm_cumulated)
    
    np.save('final_arrays/Si/diff_cs_plane_norm_' + kind + '.npy', diff_cs_Si_plane_norm)
    np.save('final_arrays/Si/diff_cs_plane_norm_cumulated_' + kind + '.npy',
            diff_cs_Si_plane_norm_cumulated)


