import matplotlib.pyplot as plt
import numpy as np
from functions import plot_functions as pf
# import mapping_aktary as const_m
from mapping import mapping_harris as const_m
import importlib
pf = importlib.reload(pf)
const_m = importlib.reload(const_m)

hist_4nm = np.load('data/harris_monomer_hist_5nm.npy')

# %%
font_size = 8
# font_size = 14

_, ax = plt.subplots(dpi=300)
# fig = plt.figure(figsize=[3.35, 3], dpi=300)
fig = plt.gcf()
fig.set_size_inches(3, 3)

hist_x = np.average(hist_4nm, axis=0)
hist_y = np.average(hist_4nm, axis=1)
hist_z = np.average(hist_4nm, axis=2)

plt.imshow(hist_x)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=font_size)

ax.set_xlabel('$x$, nm', fontsize=font_size)
ax.set_ylabel('$y$, nm', fontsize=font_size)

plt.xticks(np.array((0, 5, 10, 15, 20)) * 2, ('0', '20', '40', '60', '80'))
plt.yticks(np.array((0, 5, 10, 15, 20)) * 2, ('0', '20', '40', '60', '80'))

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.show()

# %%
# plt.savefig('hist.eps', bbox_inches='tight')
plt.savefig('hist_new.tiff', bbox_inches='tight')
