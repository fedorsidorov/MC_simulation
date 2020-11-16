import importlib
import numpy as np
import matplotlib.pyplot as plt
from functions import scission_functions as sf

sf = importlib.reload(sf)

# %%
font_size = 8

_, ax = plt.subplots(dpi=300)
fig = plt.gcf()
fig.set_size_inches(3, 3)

EE = np.linspace(0, 10, 1000)

stairway_1 = sf.get_scission_probs({"C-C2": 4}, EE)
stairway_2 = sf.get_scission_probs({"C-C2": 4, "C-C'": 2}, EE)

plt.plot(EE, stairway_1, label='C-C')
plt.plot(EE, stairway_2, label='C-C + C-ester')

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.xlabel(r'incident electron energy, eV', fontsize=font_size)
plt.ylabel(r'scission probability', fontsize=font_size)

plt.xlim(0, 10)

plt.legend(fontsize=font_size)

plt.grid()
# plt.show()

# %%
# plt.savefig('stairway.eps', bbox_inches='tight')
plt.savefig('stairway.tiff', bbox_inches='tight')
