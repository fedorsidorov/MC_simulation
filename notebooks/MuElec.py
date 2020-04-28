import numpy as np


#%%
u = np.load('data/MuElec/u_6osc.npy')
diff = np.load('data/MuElec/sigmadiff_6.npy')

# %%
diff_norm = np.zeros(np.shape(diff))

for i in range(np.shape(diff)[0]):
    for j in range(np.shape(diff)[1]):

        if np.sum(diff[i, j, :]) != 0:
            diff_norm[i, j, :] = diff[i, j, :] / np.sum(diff[i, j, :])

# %%
# np.save('Resources/MuElec/Si_MuElec_IIMFP.npy', u)
# np.save('Resources/MuElec/Si_MuElec_DIIMFP_norm.npy', diff_norm)
