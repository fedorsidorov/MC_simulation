import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# font = {'size': 14}
# matplotlib.rc('font', **font)


# %%
tt = np.load('notebooks/heating/tt.npy')
TT = np.load('notebooks/heating/T_total.npy')
err_total = np.load('notebooks/heating/err_total.npy')

plt.figure(dpi=300, figsize=[4, 3])

plt.plot(tt, TT, 'k', label=r'$\Delta T$')
plt.plot(tt, TT + err_total, 'C3', label=r'макс. $\Delta T$')
plt.plot(tt, TT - err_total, 'C0', label=r'мин. $\Delta T$')
plt.plot(tt, TT, 'k')

plt.xlim(0, 100)
plt.ylim(0, 0.8)

plt.yticks([0, 0.2, 0.4, 0.6, 0.8])

plt.xlabel(r'$t$, с')
plt.ylabel(r'$\Delta T$, °C')

plt.legend()
plt.grid()

plt.savefig('heating.jpg', dpi=300, bbox_inches='tight')
# plt.savefig('heating_14.jpg', dpi=300, bbox_inches='tight')
# plt.show()

