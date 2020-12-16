import numpy as np
import matplotlib.pyplot as plt

vlist = np.loadtxt('notebooks/SE/vlist.txt')

# vlist = vlist[np.where(vlist[:, 2] != -100)]

plt.figure(dpi=300)
plt.plot(vlist[:, 1], vlist[:, 2], '.')
plt.show()
