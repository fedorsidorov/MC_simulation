import matplotlib.pyplot as plt
import numpy as np


# %%
P1 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/1.csv', delimiter=',', skiprows=5)
P2 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/2_1.csv', delimiter=',', skiprows=5)
P3 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/3_1.csv', delimiter=',', skiprows=5)
P4 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/4_1.csv', delimiter=',', skiprows=5)
P5 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/5_1.csv', delimiter=',', skiprows=5)
P6 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/6_1.csv', delimiter=',', skiprows=5)
P7 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/7_1.csv', delimiter=',', skiprows=5)
P8 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/8_1.csv', delimiter=',', skiprows=5)
P9 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/9_1.csv', delimiter=',', skiprows=5)
P10 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/10_2_no_angle.csv', delimiter=',', skiprows=5)
P11 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/11_1_no_angle.csv', delimiter=',', skiprows=5)
P12 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/12_1.csv', delimiter=',', skiprows=5)
P13 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/13_1.csv', delimiter=',', skiprows=5)
P14 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/14_1.csv', delimiter=',', skiprows=5)
P15 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/15_1.csv', delimiter=',', skiprows=5)
P16 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/16_1.csv', delimiter=',', skiprows=5)
P17 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/17_1.csv', delimiter=',', skiprows=5)
P18 = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/18_1.csv', delimiter=',', skiprows=5)

P0 = np.loadtxt('notebooks/DEBER_profiles/Fedor/365/3_2a.csv', delimiter=',', skiprows=5)

plt.figure(dpi=300)

# plt.plot(P1[:, 0], P1[:, 1] - np.min(P1[:, 1]), label='P1')
# plt.plot(P2[:, 0], P2[:, 1] - np.min(P2[:, 1]), label='P2')
# plt.plot(P3[:, 0], P3[:, 1] - np.min(P3[:, 1]), label='P3')
# plt.plot(P4[:, 0], P4[:, 1] - np.min(P4[:, 1]), label='P4')
# plt.plot(P5[:, 0], P5[:, 1] - np.min(P5[:, 1]), label='P5')
# plt.plot(P6[:, 0], P6[:, 1] - np.min(P6[:, 1]), label='P6')
# plt.plot(P7[:, 0], P7[:, 1] - np.min(P7[:, 1]), label='P7')
# plt.plot(P8[:, 0], P8[:, 1] - np.min(P8[:, 1]), label='P8')
# plt.plot(P9[:, 0], P9[:, 1] - np.min(P9[:, 1]), label='P9')
plt.plot(P10[:, 0] / 1000 - 15.1, P10[:, 1] - np.min(P10[:, 1]) + 40, label='P10')
# plt.plot(P11[:, 0] / 1000 - 30, P11[:, 1] - np.min(P11[:, 1]), label='P11')
# plt.plot(P12[:, 0], P12[:, 1] - np.min(P12[:, 1]), label='P12')
# plt.plot(P13[:, 0], P13[:, 1] - np.min(P13[:, 1]), label='P13')
# plt.plot(P14[:, 0], P14[:, 1] - np.min(P14[:, 1]), label='P14')
# plt.plot(P15[:, 0], P15[:, 1] - np.min(P15[:, 1]), label='P15')
# plt.plot(P16[:, 0], P16[:, 1] - np.min(P16[:, 1]), label='P16')
# plt.plot(P17[:, 0], P17[:, 1] - np.min(P17[:, 1]), label='P17')
# plt.plot(P18[:, 0], P18[:, 1] - np.min(P18[:, 1]), label='P18')

plt.plot(P0[:, 0] / 1000 - 20.02, P0[:, 1] - np.min(P0[:, 1]) + 15, label='P0')

plt.legend()

plt.xlim(-2, 2)
plt.ylim(0, 160)

plt.xlabel('x, um')
plt.ylabel('z, nm')

plt.grid()
plt.show()
# plt.savefig('P11.jpg', dpi=300)

