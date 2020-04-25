#%%
import scipy.constants as const


#%% general
pi = const.pi
Na = const.N_A

#%% electron
m = const.m_e * 1e+3
e = const.e * 10 * const.c

#%% relativity
c = const.c * 1e+2

#%% quantum physics
eV = const.e * 1e+7
hbar = const.hbar * 1e+7
a0 = const.physical_constants['Bohr radius'][0]  # Bohr radius
r0 = const.physical_constants['classical electron radius'][0] * 1e+2  # classical electron radius
Ry = const.physical_constants['Rydberg constant times hc in eV'][0] * eV

#%% PMMA
rho_PMMA = 1.19
Z_H, u_H = 1, 1.008
Z_C, u_C = 6, 12.0096
Z_O, u_O = 8, 15.99903
N_H_MMA, N_C_MMA, N_O_MMA = 8, 5, 2
u_MMA = N_H_MMA*u_H + N_C_MMA*u_C + N_O_MMA*u_O
n_MMA = rho_PMMA * Na/u_MMA
m_MMA = u_MMA / Na
M0 = u_MMA / Na

K_occup = 2
K_Ebind_C = 284.2  # devera2011.pdf
K_Ebind_O = 543.1  # devera2011.pdf

Wf_PMMA = 4.68  # dapor2015.pdf

#%% Si
Z_Si = 14
u_Si = 28.086
rho_Si = 2.33
n_Si = rho_Si * Na/u_Si

#               plasm    3p     3s      2p      2s      1s
Si_MuElec_Eb = [16.65, 6.52, 13.63, 107.98, 151.55, 1828.5]
Si_MuElec_occup = [4, 2, 2, 6, 2, 2]




