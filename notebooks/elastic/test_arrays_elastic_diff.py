import numpy as np
import matplotlib.pyplot as plt
import importlib

import arrays_nm as arr

arr = importlib.reload(arr)

# %%
EE = arr.EE
theta = arr.THETA_deg
# ans = arr.PMMA_el_DIMFP_norm

# ans = arr.Si_el_DIMFP_norm
ans = np.load('Resources/MuElec/elastic_diff_sigma_sin_norm.npy')

# %%
plt.figure(dpi=300)

for i in [613, 681, 750]:

    now_ans = ans[i, :]

    now_ans_sum = np.zeros(len(now_ans))

    for j in range(1, len(now_ans)):
        now_ans_sum[j] = np.sum(now_ans[:j+1])

    print(now_ans_sum[-1])

    plt.plot(theta, now_ans_sum, label=str(EE[i]))


d_500 = np.loadtxt('notebooks/elastic/curves/Dapor_Si_500eV.txt')
d_1000 = np.loadtxt('notebooks/elastic/curves/Dapor_Si_1keV.txt')
d_2000 = np.loadtxt('notebooks/elastic/curves/Dapor_Si_2keV.txt')

plt.plot(d_500[:, 0], d_500[:, 1], '.')
plt.plot(d_1000[:, 0], d_1000[:, 1], '.')
plt.plot(d_2000[:, 0], d_2000[:, 1], '.')

plt.xlim(0, 180)
plt.ylim(0, 1)

plt.legend()
plt.grid()
plt.show()



