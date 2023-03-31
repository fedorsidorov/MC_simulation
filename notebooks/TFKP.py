import numpy as np
import matplotlib.pyplot as plt


# %%
tt = np.linspace(-100, 100, 1000)
# tt = np.array([-2, -1, 0, 1, 2])

abs_arr = (tt**2 + np.pi**2)**(1/4)


def get_uu_vv():

    uu = np.zeros(len(tt))
    vv = np.zeros(len(tt))

    for i in range(len(tt)):
        if tt[i] == 0:
            uu[i] = np.sqrt(np.pi) * np.sqrt(2) / 2
            vv[i] = np.sqrt(np.pi) * np.sqrt(2) / 2
        elif tt[i] < 0:
            uu[i] = abs_arr[i] * np.cos((np.arctan(np.pi / tt[i]) + np.pi) / 2)
            vv[i] = abs_arr[i] * np.sin((np.arctan(np.pi / tt[i]) + np.pi) / 2)
        else:
            uu[i] = abs_arr[i] * np.cos(np.arctan(np.pi / tt[i]) / 2)
            vv[i] = abs_arr[i] * np.sin(np.arctan(np.pi / tt[i]) / 2)

    return uu, vv


u_arr, v_arr = get_uu_vv()

plt.figure(dpi=300)
plt.plot(u_arr, v_arr, 'o-')

plt.grid()
plt.xlim(-1, 10)
plt.ylim(-1, 10)

plt.show()






