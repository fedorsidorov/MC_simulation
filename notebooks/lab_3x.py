import numpy as np
import matplotlib.pyplot as plt

# %%
h_l_1 = np.array([62, 60])
h_p_1 = np.array([66, 68])

h_l_2 = np.array([56, 54])
h_p_2 = np.array([74, 76])

p_1 = 13600 * 9.8 * (h_p_1 - h_l_1) * 1e-3
p_2 = 13600 * 9.8 * (h_p_2 - h_l_2) * 1e-3

T_1 = np.array([60, 73.1]) + 273
T_2 = np.array([99.6, 104]) + 273

ln_p_1 = np.log(p_1)
ln_p_2 = np.log(p_2)

T_inv_1 = 1 / T_1
T_inv_2 = 1 / T_2

delta_p = 133.3
delta_ln_p_1 = delta_p / p_1
delta_ln_p_2 = delta_p / p_2

# %%
plt.figure(dpi=300)
plt.plot(T_inv_1, ln_p_1, 'o--')
plt.plot(T_inv_1, ln_p_1 + delta_ln_p_1, 'o')
plt.plot(T_inv_1, ln_p_1 - delta_ln_p_1, 'o')

plt.plot(T_inv_2, ln_p_2, 'o--')
plt.plot(T_inv_2, ln_p_2 + delta_ln_p_2, 'o')
plt.plot(T_inv_2, ln_p_2 - delta_ln_p_2, 'o')

plt.grid()

plt.xlabel('1/T')
plt.ylabel('ln p')

plt.show()

