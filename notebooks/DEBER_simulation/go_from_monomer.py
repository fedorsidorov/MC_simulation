import importlib
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_5um_900nm as mm
from functions import array_functions as af
from functions import MC_functions as mcf
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import reflow_functions as rf
from functions import SE_functions as ef
from functions import e_beam_MC as eb_MC
from functions import diffusion_functions as df

import indexes as ind

const = importlib.reload(const)
eb_MC = importlib.reload(eb_MC)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
af = importlib.reload(af)
df = importlib.reload(df)
ef = importlib.reload(ef)
sf = importlib.reload(sf)
rf = importlib.reload(rf)

# %%
zip_length = 1000

# for i in range(47):
i = 0

now_scission_matrix = np.load('notebooks/DEBER_simulation/vary_zz_vac_0p2_5s/scission_matrix_' + str(i) + '.npy')

plt.figure(dpi=300)
plt.imshow(now_scission_matrix.transpose())
plt.show()

monomer_matrix_0 = now_scission_matrix * zip_length

# %%
z0_arr = np.ones(10000000) * 50

z_arr = df.get_final_z_arr(z0_arr, d_PMMA=100, D=1, delta_t=10000)

# %%
z_hist, bins = np.histogram(z_arr, bins=np.linspace(0, 150, 151))
centers = (bins[:-1] + bins[1:]) / 2

plt.figure(dpi=300)
plt.plot(centers, z_hist)
plt.show()









