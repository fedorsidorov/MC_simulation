import importlib
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import grid as grid
import gryzinski as gryz
import constants as c
from _outdated import arrays as a
from functions import MC_functions as u

a = importlib.reload(a)
c = importlib.reload(c)
grid = importlib.reload(grid)
gryz = importlib.reload(gryz)
u = importlib.reload(u)

# %%
Si_gryz_IIMFP = np.zeros((len(grid.EE), 5))

for n in trange(5, position=0):
    for i, E in enumerate(grid.EE):
        Si_gryz_IIMFP[i, n] = gryz.get_Gryzinski_IIMFP(
            Eb=c.Si_MuElec_E_bind[n+1],
            E=E,
            conc=c.n_Si,
            n_el=c.Si_MuElec_occup[n+1]
        )

# %%
plt.figure(dpi=300)

for n in range(5):
    plt.loglog(grid.EE, Si_gryz_IIMFP[:, n], label='simple_Si_MC e-e ' + str(n))

plt.grid()
plt.show()

# n_Si = rho_Si * Na / u_Si
#
#               plasm    3p     3s      2p      2s      1s
# Si_MuElec_E_bind = [0, 6.52, 13.63, 107.98, 151.55, 1828.5]
# Si_MuElec_E_plasmon = 16.65
# Si_MuElec_occup = [4, 2, 2, 6, 2, 2]

# %%
np.save('notebooks/Gryzinski/Si_Gryzinski_IMFP_5.npy', Si_gryz_IIMFP)
