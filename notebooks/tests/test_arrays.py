import numpy as np
import arrays_nm as arr
import matplotlib.pyplot as plt

# %%
plt.figure(dpi=300)
plt.loglog(arr.EE, 1 / arr.PMMA_val_IMFP)
plt.loglog(arr.EE, 1 / arr.C_K_ee_IMFP)
plt.loglog(arr.EE, 1 / arr.O_K_ee_IMFP)
plt.plot(arr.EE, 1 / arr.PMMA_total_IMFP)
plt.grid()
# plt.xlim(1e+1, 1e+4)
# plt.ylim(0, 100)
plt.show()
