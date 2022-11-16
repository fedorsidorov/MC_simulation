import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mapping import mapping_3p3um_80nm as mm
from functions import MC_functions as mcf
from functions import diffusion_functions as df
import constants as const
import copy
from scipy import special

const = importlib.reload(const)
mcf = importlib.reload(mcf)
df = importlib.reload(df)
mm = importlib.reload(mm)


# %%
L = 0.1
lamda = 46
C = 460
rho = 7800
T0 = 20
T_l = 300
T_r = 100
delta_t = 60

# D = lamda / rho / C
D = 1e-7
# D = 1e-9

t_end = total_time = 10
# t_end = total_time = 60000
tau = t_end / 100

N = 100
h = L / (N-1)
n0_arr = np.ones(100) * 20

n_final_arr = df.get_concentration_1d_arr_bnd1(n0_arr, T_l, T_r, D, tau, h, total_time)
# n_final_arr = get_concentration_1d_arr_bnd2_0(n0_arr, D, tau, h, total_time)
# n_final_arr = get_concentration_1d_arr_bnd_12_0(n0_arr, T_r, D, tau, h, total_time)

plt.figure(dpi=300)
plt.plot(n_final_arr)
plt.xlim(0, 100)
plt.ylim(0, 300)
plt.show()
