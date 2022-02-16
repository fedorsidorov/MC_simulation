import numpy as np
import importlib
import indexes as ind

ind = importlib.reload(ind)


# %%
dE_total = 0
n_val_total = 0

for i in range(7, 25):
    print(i)

    for j in range(10):

        now_e_DATA = np.load('/Volumes/Transcend/val_matrix/now_e_DATA_' + str(i) + '_' + str(j) + '.npy')

        now_P_e_DATA = now_e_DATA[np.where(now_e_DATA[:, ind.e_DATA_layer_id_ind] == ind.PMMA_ind)]

        now_Pv_e_DATA = now_e_DATA[np.where(
            np.logical_and(
                now_e_DATA[:, ind.e_DATA_layer_id_ind] == ind.PMMA_ind,
                now_e_DATA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind))
        ]

        dE_total += np.sum(now_P_e_DATA[:, 7])
        n_val_total += len(now_Pv_e_DATA)

# %% 2.55
G_value_th = n_val_total * 0.07 / (dE_total / 100)  # theory
G_value_sim = n_val_total * 0.09 / (dE_total / 100)  # simulation

