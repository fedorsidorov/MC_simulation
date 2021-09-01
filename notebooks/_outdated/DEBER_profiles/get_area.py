import numpy as np
import matplotlib.pyplot as plt

# %%
# profile = np.loadtxt('data/DEBER_profiles/Camscan_900nm/Camscan_0.05.txt')  # A = 381526 nm^2
# inds = range(100, 154)
# xx = profile[inds, 0] * 1000
# yy = profile[inds, 1] * 1000

# profile = np.loadtxt('data/DEBER_profiles/Camscan_900nm/Camscan_0.2.txt')  # A = 1275148 nm^2
# inds = range(90, 185)
# xx = profile[inds, 0] * 1000
# yy = profile[inds, 1] * 1000

profile = np.loadtxt('data/DEBER_profiles/Camscan_900nm/Camscan_0.87.txt')  # A = 2037395 nm^2
inds = range(120, 253)
xx = profile[inds, 0] * 1000
yy = profile[inds, 1] * 1000

plt.figure(dpi=300)
plt.plot(xx, yy, 'o-')
plt.show()

# %%
k = (yy[-1] - yy[0]) / (xx[-1] - xx[0])
yy_corr = yy - xx * k

plt.figure(dpi=300)
plt.plot(xx, yy_corr, 'o-')
plt.show()

# %%
yy_inv = yy_corr.max() - yy_corr

plt.figure(dpi=300)
plt.plot(xx, yy_inv)
plt.show()

area = np.trapz(yy_inv, x=xx)

print(area)
