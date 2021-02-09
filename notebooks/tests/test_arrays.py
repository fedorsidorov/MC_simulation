import numpy as np
import matplotlib.pyplot as plt
import importlib

import arrays_nm as arr

arr = importlib.reload(arr)

# %%
# total IMFP
PMMA_IIMFP = np.vstack((arr.PMMA_val_IMFP, arr.C_K_ee_IMFP, arr.O_K_ee_IMFP,
                       arr.PMMA_ph_IMFP, arr.PMMA_pol_IMFP)).transpose()

PMMA_total_IIMFP = np.sum(PMMA_IIMFP, axis=1)

plt.figure(dpi=300)

# plt.semilogy(arr.EE, 1 / arr.PMMA_val_IMFP)

# plt.loglog(arr.EE, 1 / arr.C_K_ee_IMFP)
# plt.loglog(arr.EE, 1 / arr.O_K_ee_IMFP)

# plt.semilogy(arr.EE, 1 / PMMA_total_IIMFP)

# plt.loglog(arr.EE, 1 / arr.PMMA_val_IMFP + 1 / arr.C_K_ee_IMFP + 1 / arr.O_K_ee_IMFP)

plt.semilogy(arr.EE, 1 / arr.PMMA_ph_IMFP)
plt.semilogy(arr.EE, 1 / arr.PMMA_pol_IMFP)

plt.xlim(0, 200)
plt.ylim(0.5, 100)

plt.grid()
plt.show()

# %%
PMMA_el_IMFP = \
    np.load('/Users/fedor/PycharmProjects/MC_simulation/Resources/ELSEPA/PMMA/PMMA_muffin_EIMFP.npy') * 1e-7

plt.figure(dpi=300)

plt.loglog(arr.EE, PMMA_el_IMFP)

plt.show()

