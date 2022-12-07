import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %%
tau = np.load('notebooks/Boyd_kinetic_curves/tau.npy')
M1w = np.load('notebooks/Boyd_kinetic_curves/M1w_130_term.npy')
yw = np.load('notebooks/Boyd_kinetic_curves/yw_130_term.npy')
z = np.load('notebooks/Boyd_kinetic_curves/z_130_term.npy')

# %%
plt.figure(dpi=600, figsize=[4, 3])
plt.semilogy(tau, M1w, label=r'$\widetilde{M}_1$')
plt.semilogy(tau, yw, label=r'$\widetilde{y}$')
plt.ylim(1e-4, 1)
plt.xlabel(r'$\tau$')
# plt.ylabel(r'$\widetilde{M}_1$, $\widetilde{y}$')
plt.legend(fontsize=10)
plt.grid()

plt.xlim(0, 400)

# plt.savefig('M1_y_14.jpg', bbox_inches='tight')
plt.savefig('SZ_M1_y_14_noY.jpg', bbox_inches='tight')
# plt.show()

# %%
plt.figure(dpi=600, figsize=[4, 3])
plt.plot(tau, z, label='$z$')
plt.ylim(-0.3, 0)
plt.xlabel(r'$\tau$')
# plt.ylabel(r'$z$')
plt.legend(fontsize=10, loc='lower right')
plt.grid()

plt.xlim(0, 400)

# plt.savefig('z_14.jpg', bbox_inches='tight')
plt.savefig('SZ_z_14_noY.jpg', bbox_inches='tight')
plt.show()
