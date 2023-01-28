import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %%
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / 2 / sigma**2)


# %% SET 1
# loc_arr = [0, 250, 500, 750, 1000]
# p_arr = [0.5, 0.25, 0.1, 0.1, 0.05]

beam_sigma = 200

xx = np.linspace(-2000, 2000, 3001)

gauss_1 = gauss(xx, 1, 0, beam_sigma)
gauss_2 = gauss(xx, 1/3, beam_sigma * 2, beam_sigma)
gauss_3 = gauss(xx, 1/5, beam_sigma * 4, beam_sigma)
gauss_4 = gauss(xx, 1/7, beam_sigma * 6, beam_sigma)

xx_final = np.concatenate([xx - 3000, xx, xx + 3000])

gauss_1_final = np.concatenate([gauss_1, gauss_1, gauss_1])
gauss_2_final = np.concatenate([gauss_2, gauss_2, gauss_2])
gauss_3_final = np.concatenate([gauss_3, gauss_3, gauss_3])
gauss_4_final = np.concatenate([gauss_4, gauss_4, gauss_4])

plt.figure(dpi=300, figsize=[4, 3])
plt.plot(xx_final / 1000, gauss_1_final, label=r'$j_0$')
plt.plot(xx_final / 1000, gauss_2_final, label=r'$j_0/3$')
plt.plot(xx_final / 1000, gauss_3_final, label=r'$j_0/5$')
plt.plot(xx_final / 1000, gauss_4_final, label=r'$j_0/7$')

# plt.plot(xx, gauss(xx, 0.5, 0, beam_sigma), label=r'$x_0 = 0$ нм')
# plt.plot(xx, gauss(xx, 0.25, 250, beam_sigma), label=r'$x_0 = 250$ нм')
# plt.plot(xx, gauss(xx, 0.1, 500, beam_sigma), label=r'$x_0 = 500$ нм')
# plt.plot(xx, gauss(xx, 0.1, 750, beam_sigma), label=r'$x_0 = 750$ нм')
# plt.plot(xx, gauss(xx, 0.05, 1000, beam_sigma), label=r'$x_0 = 1000$ нм')

# plt.plot(xx_final, gauss_1_final, label=r'$x_0 = 0$ нм')
# plt.plot(xx_final, gauss_2_final, label=r'$x_0 = 250$ нм')
# plt.plot(xx_final, gauss_3_final, label=r'$x_0 = 500$ нм')
# plt.plot(xx_final, gauss_4_final, label=r'$x_0 = 750$ нм')
# plt.plot(xx_final, gauss_5_final, label=r'$x_0 = 1000$ нм')

plt.legend(fontsize=10, loc='upper right')

plt.xlabel(r'$x$, мкм')
# plt.ylabel(r'$\frac{j_{beam}}{j_0}$')
plt.ylabel(r'$j_{beam} / j_0$')

# plt.xlim(-4, 4)
plt.xlim(-5, 5)
# plt.ylim(0, 1.2)
plt.ylim(0, 1.25)

plt.grid()

plt.savefig('asymmetric_beam.jpg', dpi=300, bbox_inches='tight')
plt.show()


# %% SET 2
beam_sigma = 200

xx = np.linspace(-1500, 1500, 3001)

gauss_1 = gauss(xx, 1, 400, beam_sigma)
gauss_2 = gauss(xx, 1, -400, beam_sigma)

xx_final = np.concatenate([xx - 3000, xx, xx + 3000])

gauss_1_final = np.concatenate([gauss_1, gauss_1, gauss_1])
gauss_2_final = np.concatenate([gauss_2, gauss_2, gauss_2])

plt.figure(dpi=300, figsize=[4, 3])
plt.plot(xx_final / 1000, gauss_1_final, label=r'$j_0$')
plt.plot(xx_final / 1000, gauss_2_final, label=r'$j_0$')

# plt.plot(xx, gauss(xx, 0.5, 0, beam_sigma), label=r'$x_0 = 0$ нм')
# plt.plot(xx, gauss(xx, 0.25, 250, beam_sigma), label=r'$x_0 = 250$ нм')
# plt.plot(xx, gauss(xx, 0.1, 500, beam_sigma), label=r'$x_0 = 500$ нм')
# plt.plot(xx, gauss(xx, 0.1, 750, beam_sigma), label=r'$x_0 = 750$ нм')
# plt.plot(xx, gauss(xx, 0.05, 1000, beam_sigma), label=r'$x_0 = 1000$ нм')

# plt.plot(xx_final, gauss_1_final, label=r'$x_0 = 0$ нм')
# plt.plot(xx_final, gauss_2_final, label=r'$x_0 = 250$ нм')
# plt.plot(xx_final, gauss_3_final, label=r'$x_0 = 500$ нм')
# plt.plot(xx_final, gauss_4_final, label=r'$x_0 = 750$ нм')
# plt.plot(xx_final, gauss_5_final, label=r'$x_0 = 1000$ нм')

plt.legend(fontsize=10, loc='upper right')

plt.xlabel(r'$x$, мкм')
plt.ylabel(r'$j_{beam}$')

plt.xlim(-5, 5)
plt.ylim(0, 1.2)

plt.grid()

plt.savefig('pm_400_colors.jpg', dpi=300, bbox_inches='tight')
plt.show()
