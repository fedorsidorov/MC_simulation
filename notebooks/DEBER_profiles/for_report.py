import numpy as np
import matplotlib.pyplot as plt

# %% 361
pr_DEBER = np.loadtxt('notebooks/DEBER_profiles/Fedor/361/D_dark_1/D1_2.csv', delimiter=',', skiprows=5)
pr_WET = np.loadtxt('notebooks/DEBER_profiles/Fedor/7_many/10_2_no_angle.csv', delimiter=',', skiprows=5)

fig, ax_1 = plt.subplots(dpi=600)
fig.set_size_inches(4*1.5, 3*1.5)

ax_1.plot(pr_WET[:, 0] / 1000 - 14.8, pr_WET[:, 1] - np.min(pr_WET[:, 1]) + 40, color='C0', label='мокрая литография')

plt.legend(loc=2)
plt.xlabel('x, мкм')
plt.grid()

ax_2 = ax_1.twinx()
ax_2.plot(pr_DEBER[:, 0] / 1000, pr_DEBER[:, 1] - np.min(pr_DEBER[:, 1]) + 100, color='C1', label='терм. усиленный резист')


ax_1.set_ylabel('z, нм')
ax_2.set_ylabel('z, нм')

# for tick in ax_1.yaxis.get_major_ticks():
#     tick.label.set_color('red')
#
# for tick in ax_2.yaxis.get_major_ticks():
#     tick.label.set_color('C0')

plt.legend(loc=0)

plt.xlim(0, 10)

plt.show()
# plt.savefig('for_report.jpg', dpi=600)
