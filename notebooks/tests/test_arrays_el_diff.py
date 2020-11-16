import numpy as np
import arrays_nm as arr
import matplotlib.pyplot as plt


# %%
EE = arr.EE
theta = arr.THETA_deg
ans = arr.PMMA_el_DIMFP_norm

# %%
plt.figure(dpi=300)

for i in range(0, len(EE), 100):
    plt.plot(theta, ans[i, :], label=str(EE[i]))

plt.legend()
plt.show()



