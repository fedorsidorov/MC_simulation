import numpy as np
import matplotlib.pyplot as plt

# %%
prof = np.loadtxt('data/DEBER_profiles/Camscan_80nm/Camscan_new.txt')

inds = range(42, 160)

xx = prof[inds, 0]
yy = prof[inds, 1]

plt.figure(dpi=300)
plt.plot(xx, yy, 'o-')
plt.show()

# %%
k = (yy[-1] - yy[0]) / (xx[-1] - xx[0])
yy_corr = yy - xx * k

plt.figure(dpi=300)
plt.plot(xx, yy_corr)
plt.show()

# %%
yy_inv = yy_corr.max() - yy_corr

plt.figure(dpi=300)
plt.plot(xx, yy_inv)
plt.show()

area = np.trapz(yy_inv, x=xx)

print(area)
