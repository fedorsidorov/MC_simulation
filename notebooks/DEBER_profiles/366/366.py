import numpy as np
import matplotlib.pyplot as plt

# %%
D1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/2.csv', delimiter=',', skiprows=5)
D3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/3.csv', delimiter=',', skiprows=5)
D4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/4.csv', delimiter=',', skiprows=5)
D5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/5.csv', delimiter=',', skiprows=5)
D6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/6.csv', delimiter=',', skiprows=5)
D7 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/7.csv', delimiter=',', skiprows=5)
D8 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/8.csv', delimiter=',', skiprows=5)
D9 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/9.csv', delimiter=',', skiprows=5)
D10 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/10.csv', delimiter=',', skiprows=5)
D11 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/11.csv', delimiter=',', skiprows=5)
D12 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/12.csv', delimiter=',', skiprows=5)
D13 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/13.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(D1[:, 0], D1[:, 1] - np.min(D1[:, 1]), label='D1')
plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')
plt.plot(D3[:, 0], D3[:, 1] - np.min(D3[:, 1]), label='D3')
plt.plot(D4[:, 0], D4[:, 1] - np.min(D4[:, 1]), label='D4')
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

# plt.plot(xx[320:422], zz[320:422], label='366 D11 real')

plt.legend()

plt.xlim(-5000, 5000)
# plt.xlim(-5, 5)
plt.ylim(0, 600)

plt.xlabel(r'x, $\mu$m')
plt.ylabel('z, nm')

plt.grid()
plt.show()

# %% slice_1 frame
D1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/2.csv', delimiter=',', skiprows=5)
D3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/3.csv', delimiter=',', skiprows=5)
D4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/4.csv', delimiter=',', skiprows=5)
D5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/5.csv', delimiter=',', skiprows=5)
D6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/6.csv', delimiter=',', skiprows=5)
D7 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/7.csv', delimiter=',', skiprows=5)
D8 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/8.csv', delimiter=',', skiprows=5)
D9 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/9.csv', delimiter=',', skiprows=5)
D10 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/10.csv', delimiter=',', skiprows=5)
D11 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/11.csv', delimiter=',', skiprows=5)
D12 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/12.csv', delimiter=',', skiprows=5)
D13 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/13.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

# plt.plot(D1[:, 0], D1[:, 1] - np.min(D1[:, 1]), label='D1')
# plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')
plt.plot(D3[:, 0], D3[:, 1] - np.min(D3[:, 1]), label='D3')
plt.plot(D4[:, 0], D4[:, 1] - np.min(D4[:, 1]), label='D4')
# plt.plot(D5[:, 0], D5[:, 1] - np.min(D5[:, 1]), label='D5')
# plt.plot(D6[:, 0], D6[:, 1] - np.min(D6[:, 1]), label='D6')
# plt.plot(D7[:, 0], D7[:, 1] - np.min(D7[:, 1]), label='D7')
# plt.plot(D8[:, 0], D8[:, 1] - np.min(D8[:, 1]), label='D8')
plt.plot(D9[:, 0], D9[:, 1] - np.min(D9[:, 1]), label='D9')
plt.plot(D10[:, 0], D10[:, 1] - np.min(D10[:, 1]), label='D10')
plt.plot(D11[:, 0], D11[:, 1] - np.min(D11[:, 1]), label='D11')
plt.plot(D12[:, 0], D12[:, 1] - np.min(D12[:, 1]), label='D12')
plt.plot(D13[:, 0], D13[:, 1] - np.min(D13[:, 1]), label='D13')

plt.legend()

# plt.xlim(0, 10000)

plt.grid()
plt.show()

# %% slice_1
D1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice/1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice/2.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(D1[:, 0], D1[:, 1] - np.min(D1[:, 1]), label='D1')
plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')

plt.legend()

# plt.xlim(0, 10000)

plt.grid()
plt.show()

# %% slice_frame
D1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/1.csv', delimiter=',', skiprows=5)
D2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/2.csv', delimiter=',', skiprows=5)
D3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/3.csv', delimiter=',', skiprows=5)
D4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/4.csv', delimiter=',', skiprows=5)
D5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/5.csv', delimiter=',', skiprows=5)
D6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/6.csv', delimiter=',', skiprows=5)
D7 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/7.csv', delimiter=',', skiprows=5)
D8 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/8.csv', delimiter=',', skiprows=5)
D9 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/9.csv', delimiter=',', skiprows=5)
D10 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/10.csv', delimiter=',', skiprows=5)
D11 = np.loadtxt('notebooks/DEBER_profiles/Fedor/366/slice_frame/11.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

plt.plot(D1[:, 0], D1[:, 1] - np.min(D1[:, 1]), label='D1')
plt.plot(D2[:, 0], D2[:, 1] - np.min(D2[:, 1]), label='D2')
plt.plot(D3[:, 0], D3[:, 1] - np.min(D3[:, 1]), label='D2')
plt.plot(D4[:, 0], D4[:, 1] - np.min(D4[:, 1]), label='D2')
plt.plot(D5[:, 0], D5[:, 1] - np.min(D5[:, 1]), label='D2')
plt.plot(D6[:, 0], D6[:, 1] - np.min(D6[:, 1]), label='D2')
plt.plot(D7[:, 0], D7[:, 1] - np.min(D7[:, 1]), label='D2')
plt.plot(D8[:, 0], D8[:, 1] - np.min(D8[:, 1]), label='D2')
plt.plot(D9[:, 0], D9[:, 1] - np.min(D9[:, 1]), label='D2')
plt.plot(D10[:, 0], D10[:, 1] - np.min(D10[:, 1]), label='D2')

plt.legend()

# plt.xlim(0, 10000)

plt.grid()
plt.show()

