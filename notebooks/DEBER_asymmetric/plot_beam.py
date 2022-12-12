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

beam_sigma = 250

xx = np.linspace(-2000, 2000, 20001)

gauss_1 = gauss(xx, 0.5, 0, beam_sigma)
gauss_2 = gauss(xx, 0.25, 250, beam_sigma)
gauss_3 = gauss(xx, 0.1, 500, beam_sigma)
gauss_4 = gauss(xx, 0.1, 750, beam_sigma)
gauss_5 = gauss(xx, 0.05, 1000, beam_sigma)

xx_final = np.concatenate([xx - 3000, xx, xx + 3000])

gauss_1_final = np.concatenate([gauss_1, gauss_1, gauss_1])
gauss_2_final = np.concatenate([gauss_2, gauss_2, gauss_2])
gauss_3_final = np.concatenate([gauss_3, gauss_3, gauss_3])
gauss_4_final = np.concatenate([gauss_4, gauss_4, gauss_4])
gauss_5_final = np.concatenate([gauss_5, gauss_5, gauss_5])


fig, ax = plt.figure(dpi=600, figsize=[4, 3])
plt.plot(xx_final, gauss_1_final, label=r'50 %')
plt.plot(xx_final, gauss_2_final, label=r'25 %')
plt.plot(xx_final, gauss_3_final, label=r'10 %')
plt.plot(xx_final, gauss_4_final, label=r'10 %')
plt.plot(xx_final, gauss_5_final, label=r'5 %')

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

plt.xlabel(r'$x$, нм')
plt.ylabel(r'интенсивность')

plt.xlim(-5000, 5000)
plt.ylim(0, 0.6)

plt.grid()

# plt.savefig('asymmetric_1.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %% SET 2
# loc_arr = np.array([0, 250, 500, 750])
# p_arr = [0.5, 0.25, 0.15, 0.1]




