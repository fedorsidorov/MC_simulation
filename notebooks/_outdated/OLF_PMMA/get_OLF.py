import numpy as np
import matplotlib.pyplot as plt


# %%
r_Im = np.loadtxt('data/Dapor/Ritsko_dashed.txt')
r_Im_1 = np.loadtxt('data/Dapor/Ritsko_Im.txt')

plt.figure(dpi=300)
plt.loglog(r_Im[:, 0], r_Im[:, 1], 'ro')
# plt.loglog(r_Im_1[:, 0], r_Im_1[:, 1], 'bo')
plt.show()




