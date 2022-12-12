import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}
matplotlib.rc('font', **font)


# %%
TT = np.array([150,
               149, 148, 147, 146, 145, 144, 143, 142, 141, 140,
               139, 138, 137, 136, 135, 134, 133, 132, 131, 130,
               129, 128, 127, 126, 125, 124, 123, 122, 121, 120,
               119, 118, 117, 116, 115, 114, 113, 112, 111, 110,
               109, 108, 107, 106, 105, 104, 103, 102, 101, 100,
               99, 98, 97, 96, 95, 94, 93, 92, 91, 90,
               89, 88, 87, 86, 85, 84, 83, 82, 81, 80
               ])

tt = np.array([8,
               4, 4, 3, 2, 5, 2, 4, 3, 3, 3,
               4, 2, 4, 3, 3, 3, 4, 3, 3, 4,
               3, 3, 4, 4, 3, 3, 4, 4, 4, 4,
               3, 4, 4, 5, 4, 4, 4, 5, 4, 4,
               5, 5, 4, 6, 4, 5, 5, 5, 5, 5,
               6, 6, 5, 6, 6, 5, 6, 6, 6, 7,
               7, 6, 7, 6, 8, 7, 7, 6, 9, 9
               ])

tt_sum = np.zeros(len(TT))

for i in range(len(tt_sum)):
    tt_sum[i] = np.sum(tt[:i])


plt.figure(dpi=600, figsize=[4, 3])
plt.plot(tt_sum, TT)
plt.xlabel(r'$t$, с')
plt.ylabel(r'$T$, $^{\circ}C$')

plt.xlim(0, 300)
# plt.ylim(0, 175)
plt.ylim(50, 200)
plt.grid()

plt.savefig('cooling.jpg', dpi=600, bbox_inches='tight')
plt.show()









