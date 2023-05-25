import numpy as np
import matplotlib.pyplot as plt


# %%
tau = np.load('notebooks/Boyd_Schulz_Zimm/tau.npy')

MN = np.load('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_150_term.npy')
MN_150 = np.load('notebooks/Boyd_Schulz_Zimm/for_vary_T/Mn_150_term_150.npy')

plt.figure(dpi=300)
plt.semilogy(tau, MN, label='zip len = 1000')
plt.semilogy(tau, MN_150, label='zip len = 150')
plt.xlabel(r'$\tau$ = $y_0 k_S t$')
plt.ylabel(r'$M$n')
plt.legend()
plt.grid()
plt.show()









