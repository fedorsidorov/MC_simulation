# %%
# import scipy.constants as const
import numpy as np

# %% general
# Na = const.N_A
Na = 6.02214076e+23

# %% electron
# m = const.m_e * 1e+3
m = 9.1093837015e-28
# e = const.e * 10 * const.c
e = 4.803204712570263e-10
e_SI = 1.6021766209e-19

# %% relativity
# c = const.c * 1e+2
c = 29979245800.0

# %% quantum physics
# eV = const.e * 1e+7
eV = 1.602176634e-12
eV_SI = eV * 1e-7
# hbar = const.hbar * 1e+7
hbar = 1.0545718176461565e-27
# a0 = const.physical_constants['Bohr radius'][0] * 1e+2  # Bohr radius
a0 = 5.2917721090299995e-09
# r0 = const.physical_constants['classical electron radius'][0] * 1e+2  # classical electron radius
r0 = 2.8179403262e-13
# Ry = const.physical_constants['Rydberg constant times hc in eV'][0] * eV
Ry = 2.1798723611035473e-11
# p_au = const.physical_constants['atomic unit of momentum'][0] * 1e+5
P_au = 1.9928519141e-19
# E_au = const.physical_constants['atomic unit of energy'][0] * 1e+7
E_au = 4.3597447222071e-11

# %% PMMA
rho_PMMA = 1.19  # g / cc
Z_H, u_H = 1, 1.008
Z_C, u_C = 6, 12.0096
Z_O, u_O = 8, 15.999
N_H_MMA, N_C_MMA, N_O_MMA = 8, 5, 2
u_MMA = N_H_MMA * u_H + N_C_MMA * u_C + N_O_MMA * u_O
n_MMA = rho_PMMA * Na / u_MMA  # 1 / cc
# m_MMA = u_MMA / Na
M_mon = u_MMA / Na  # g
V_mon_cm3 = 1 / n_MMA  # cc
V_mon_nm3 = V_mon_cm3 * 1e+21

K_occup = 2

val_E_bind_PMMA = 0
# val_E_bind_PMMA = 15  # dapor 2017.pdf

K_Ebind_C, K_Ebind_O = 284.2, 543.1  # devera2011.pdf
Zs_C, Zs_O = 5.7, 7.7

Wf_PMMA = 4.05  # PhD_thesis_Dapor.pdf
# Wf_PMMA = 4.68  # dapor2015.pdf
W_phonon = 0.1  # dapor2015.pdf

CC_bond_length = 0.28  # nm

# %% Si
Z_Si = 14
u_Si = 28.086
rho_Si = 2.33
n_Si = rho_Si * Na / u_Si

Si_val_E_bind = 15
#               plasm    3p     3s      2p      2s      1s
# Si_MuElec_E_bind = [0, 6.52, 13.63, 107.98, 151.55, 1828.5]
Si_MuElec_E_bind = [0, Si_val_E_bind, Si_val_E_bind, 107.98, 151.55, 1828.5]

# Si_MuElec_E_plasmon = 16.7
Si_MuElec_E_plasmon = 20

# Si_MuElec_occup = [4, 2, 2, 6, 2, 2]

# %%
nm3_to_cm_3 = 1e-7 ** 3
uint16_max = np.iinfo(np.uint16).max
uint32_max = np.iinfo(np.uint32).max
