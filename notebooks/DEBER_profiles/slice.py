import numpy as np
import matplotlib.pyplot as plt

# %%
slice = np.loadtxt('notebooks/DEBER_profiles/357/slice_1/1d.csv', delimiter=',', skiprows=5)
# slice = np.loadtxt('notebooks/DEBER_profiles/357/slice_1/1b.csv', delimiter=',', skiprows=5)
# slice = np.loadtxt('notebooks/DEBER_profiles/357/slice_1/1c.csv', delimiter=',', skiprows=5)
# slice = np.loadtxt('notebooks/DEBER_profiles/357/slice_1/2a.csv', delimiter=',', skiprows=5)
# slice = np.loadtxt('notebooks/DEBER_profiles/357/slice_1/3a.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)
plt.plot(slice[:, 0], slice[:, 1])

plt.ylim(250, 650)

plt.grid()
plt.show()








