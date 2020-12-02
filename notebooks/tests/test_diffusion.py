import importlib
import numpy as np
from mapping import mapping_3p3um_80nm as mapping
import matplotlib.pyplot as plt
from functions import diffusion_functions as df
import tqdm
mapping = importlib.reload(mapping)
df = importlib.reload(df)

# %%
mon_matrix = np.load('/Users/fedor/PycharmProjects/MC_simulation/new_monomer_matrix.npy')
mon_matrix_2d = np.sum(mon_matrix, axis=1)

xx = mapping.x_centers_25nm * 1e-7
zz_vac = np.zeros(len(xx))
d_PMMA = mapping.z_max * 1e-7

dT = 5
time_s = 10

# monomer_matrix_2d_final = df.track_all_monomers(mon_matrix_2d, xx, zz_vac, d_PMMA, dT, wp=1, t_step=time_s, dtdt=0.05)

progress_bar = tqdm.tqdm(total=356989, position=0)

for i in range(356989):
    now_x, now_z, total_time, history = df.track_monomer(0, 50, xx, zz_vac, d_PMMA, dT, wp=1, t_step=time_s, dtdt=0.5)
    progress_bar.update()

# plt.figure(dpi=300)
# plt.plot(history[:, 0], history[:, 1], 'o-')
# plt.show()
