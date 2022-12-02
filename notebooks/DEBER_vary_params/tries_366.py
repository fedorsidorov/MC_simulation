import numpy as np
import matplotlib.pyplot as plt

# %%
xx_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/xx_366_zero.npy')
zz_366 = np.load('notebooks/DEBER_simulation/exp_profiles/366/zz_366_zero.npy')

# folder = '/Volumes/Transcend/SIM_DEBER/150C_100s_final/sigma250_zip150/'
# folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/E_5_I_1/'
# folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/E_15_I_1/'
# folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/E_25_I_1/'

# folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/T_140/'
# folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/T_145/'
# folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/T_155/'
# folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/T_160/'

# folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/I_1/'
folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/I_1.5/'
# folder = '/Volumes/Transcend/SIM_DEBER/366_vary_params/I_2/'

xx = np.load(folder + 'try_0/xx_bins.npy')
zz = np.load(folder + 'try_0/zz_vac_bins.npy')

# plt.figure(dpi=300)
# plt.plot(XX, zz)
# plt.show()


# %%
zz_avg = np.zeros(len(zz))

zz_min = np.ones(len(xx)) * 500
zz_max = np.zeros(len(xx))

# n_tries = 100
# n_tries = 10
# n_tries = 5
n_tries = 1

# for n_try in range(1, n_tries):
for n_try in range(n_tries):
    zz_now = np.load(folder + 'try_' + str(n_try) + '/zz_vac_bins.npy')
    zz_avg += zz_now

    for j in range(len(xx)):
        if zz_now[j] < zz_min[j]:
            zz_min[j] = zz_now[j]
        if zz_now[j] > zz_max[j]:
            zz_max[j] = zz_now[j]

# zz_avg /= 99
zz_avg /= n_tries

plt.figure(dpi=300)
plt.plot(xx, zz_avg, label='average')
plt.plot(xx_366, zz_366 + 75, 'k', label='exp')
plt.plot(xx_366, zz_366 + 100, 'k', label='exp')
plt.plot(xx, zz_min, 'r--', label='MIN')
plt.plot(xx, zz_max, 'r--', label='MAX')

# plt.title('REFERENCE')
# plt.title('5 keV')
# plt.title('15 keV')
# plt.title('25 keV')

# plt.title('140 C')
# plt.title('145 C')
# plt.title('155 C')
# plt.title('160 C')

# plt.title('1 nA')
plt.title('1.5 nA')
# plt.title('2 nA')

plt.xlabel('$x$, нм')
plt.ylabel('$z$, нм')
plt.legend()
plt.grid()
plt.xlim(-1500, 1500)
# plt.ylim(-300, 600)
plt.ylim(0, 800)

# plt.savefig('REFERENCE.jpg')
# plt.savefig('5_keV.jpg')
# plt.savefig('15_keV.jpg')
# plt.savefig('25_keV.jpg')

# plt.savefig('145C.jpg')
# plt.savefig('155C.jpg')
# plt.savefig('160C.jpg')

# plt.savefig('I1.jpg')
plt.savefig('I1p5.jpg')
# plt.savefig('I2.jpg')
plt.show()



