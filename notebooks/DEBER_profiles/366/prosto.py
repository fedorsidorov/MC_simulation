import numpy as np
import matplotlib.pyplot as plt

# %%
D1 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/2.csv', delimiter=',', skiprows=5)
D3 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/3.csv', delimiter=',', skiprows=5)
D4 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/4.csv', delimiter=',', skiprows=5)
D5 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/5.csv', delimiter=',', skiprows=5)
D6 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/6.csv', delimiter=',', skiprows=5)
D7 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/7.csv', delimiter=',', skiprows=5)
D8 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/8.csv', delimiter=',', skiprows=5)
D9 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/9.csv', delimiter=',', skiprows=5)
D10 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/10.csv', delimiter=',', skiprows=5)
D11 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/11.csv', delimiter=',', skiprows=5)
D12 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/12.csv', delimiter=',', skiprows=5)
D13 = np.loadtxt('notebooks/DEBER_profiles/366/366/prosto/13.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

# plt.plot(D1[:, 0], D1[:, 1] - np.min(D1[:, 1]), label='D1')
# plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')
# plt.plot(D3[:, 0], D3[:, 1] - np.min(D3[:, 1]), label='D3')
# plt.plot(D4[:, 0], D4[:, 1] - np.min(D4[:, 1]), label='D4')
# plt.plot(D5[:, 0], D5[:, 1] - np.min(D5[:, 1]), label='D5')
# plt.plot(D6[:, 0], D6[:, 1] - np.min(D6[:, 1]), label='D6')
# plt.plot(D7[:, 0], D7[:, 1] - np.min(D7[:, 1]), label='D7')
# plt.plot(D8[:, 0], D8[:, 1] - np.min(D8[:, 1]), label='D8')
# plt.plot(D9[:, 0], D9[:, 1] - np.min(D9[:, 1]), label='D9')
# plt.plot(D10[:, 0], D10[:, 1] - np.min(D10[:, 1]), label='D10')
# plt.plot(D11[:, 0], D11[:, 1] - np.min(D11[:, 1]), label='D11')
# plt.plot(D12[:, 0], D12[:, 1] - np.min(D12[:, 1]), label='D12')
# plt.plot(D13[:, 0], D13[:, 1] - np.min(D13[:, 1]), label='D13')

xx = (D11[:, 0] / 1000 - 10 - 0.85) * 1000
zz = D11[:, 1] - np.min(D11[:, 1]) + 50

# xx = D11[:, 0] - 35000
# zz = D11[:, 1] - np.min(D11[:, 1])

# inds = np.where(np.logical_and(
#     xx >= -2000, xx <= 2000
# ))

# xx = xx[inds]
# zz = zz[inds]

xx_final = xx[310:432]
zz_final = zz[310:432] - np.min(zz[310:432])

np.save('xx_366.npy', xx_final)
np.save('zz_366_zero.npy', zz_final)

# plt.plot(xx[320:422], zz[320:422], label='366 D11 real')
# plt.plot(xx[310:432], zz[310:432], label='366 D11 real')
plt.plot(xx_final, zz_final, label='366 D11 real')
# plt.plot(xx, zz, label='366 D11 real')

plt.legend()

# plt.xlim(0, 10000)
plt.xlim(-2000, 2000)
# plt.xlim(-10000, 10000)
# plt.xlim(-5, 5)
# plt.ylim(0, 600)

plt.xlabel(r'x, $\mu$m')
plt.ylabel('z, nm')

plt.grid()
plt.show()
