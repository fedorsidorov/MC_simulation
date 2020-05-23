import importlib
import numpy as np
import matplotlib.pyplot as plt

# initial = np.load('data/chains/Harris/lens_initial.npy')
# initial = np.load('/Volumes/ELEMENTS/PyCharm_may/prepared_chains/Harris/chain_lens.npy')
initial = np.load('/Volumes/ELEMENTS/PyCharm_may/chains/Harris/lens_initial.npy')
# final_1 = np.load('/Volumes/ELEMENTS/PyCharm_may/chains/Harris/lens_final_C-C2:4.npy')
final_2 = np.load('/Volumes/ELEMENTS/PyCharm_may/chains/Harris/lens_final_C-C2:4_C-C\':2.npy')

final_1 = np.load('data/exposed_chains/Harris/harris_lens_final_4+2.npy')


font_size = 8
fig, ax = plt.subplots(dpi=300)
# fig = plt.figure(dpi=300)
# fig = plt.gcf()
fig.set_size_inches(3, 3)

bins_12 = np.linspace(2, 4, 10)

h_i, b_i = np.histogram(np.log10(initial * 100), bins=15, density=True)
h_1, b_1 = np.histogram(np.log10(final_1 * 100), bins=bins_12, density=True)
h_2, b_2 = np.histogram(np.log10(final_2 * 100), bins=bins_12, density=True)

c_i = (b_i[:-1] + b_i[1:]) / 2
c_1 = (b_1[:-1] + b_1[1:]) / 2
c_2 = (b_2[:-1] + b_2[1:]) / 2

w_1 = b_1[1:] - b_1[:-1]

plt.bar(c_i, h_i, width=0.24, label='initial')
plt.bar(b_1[:-1], h_1, width=-0.1, align='edge', label='C-C')
plt.bar(b_2[:-1], h_2, width=0.1, align='edge', label='C-C + C-ester')

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

plt.xlabel(r'log(Mw)', fontsize=font_size)
plt.ylabel(r'arbitrary units', fontsize=font_size)

plt.grid()
plt.xlim(2, 8)
plt.ylim(0, 1)
ax.legend(fontsize=font_size, loc='upper left')
plt.show()

# %%
# plt.savefig('distr.eps', bbox_inches='tight')
# plt.savefig('distr.tiff', bbox_inches='tight')
