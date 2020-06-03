import numpy as np
import matplotlib.pyplot as plt

# %%
prof = np.loadtxt('data/DEBER_profiles/Ultra_4s.txt')

xx = prof[:, 0]
yy = prof[:, 1]

k = (yy[-1] - yy[0]) / (xx[-1] - xx[0])
yy_corr = yy - xx * k

plt.figure(dpi=300)
plt.plot(xx, yy_corr)
plt.show()

# %%
yy_inv = yy_corr.max() - yy_corr

plt.figure(dpi=300)
plt.plot(xx, yy_inv)
# plt.show()

area = np.trapz(yy_inv, x=xx)

