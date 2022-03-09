import numpy as np
import matplotlib.pyplot as plt

# %%
mu_0 = 4 * np.pi * 1e-7

R = 20e-3
I_m = 70e-3
# I_eff = I_m / np.sqrt(2)
N = 2400
N_k = 500
l = 180e-3
R = 20e-3
d = 7e-3
nu = 1000

n = N / l


def get_B_theor(x):
    B_theor = mu_0 * n * I_m / 2 * (
        (l + x) / np.sqrt((l + x)**2 + R**2) - x / np.sqrt(x**2 + R**2)
    )
    return B_theor


def get_B_exp(u_eff):
    B_exp = u_eff / (((np.pi * d)**2 * N_k * nu)/(2 * np.sqrt(2)))
    return B_exp


xx = np.array([
    -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60
]) * 1e-3

U_eff = np.array([
    97.49, 98.39, 98.72, 97.99, 97.68, 95.99, 92.33, 86.83, 73.08, 50.02, 27.68, 14.13, 8.31, 4.98, 3.44, 2.58
]) * 1e-3

B_theor = get_B_theor(xx) * 1000
B_exp = get_B_exp(U_eff) * 1000

dB_exp = 0.03857
delta_B_exp = B_exp * dB_exp

plt.figure(dpi=300)
plt.errorbar(xx, B_exp, yerr=delta_B_exp, label='эксп')
plt.errorbar(xx, B_theor, label='теор')


plt.xlabel('x, мм')
plt.ylabel('B, мТл')
plt.legend()

plt.xlim(-0.1, 0.08)
plt.ylim(0, 1.4)
plt.grid()

plt.show()
# plt.savefig('lab_3_2.jpg', dpi=300)

# %%
R_T = 8.9

xx = np.array([40, 50, 60]) * 1e-3
U_T = np.array([0.230, 0.345, 0.488])

I_l = U_T / R_T

B_exp = np.array([0.058, 0.040, 0.030])

B_z = B_exp * I_l / I_m

