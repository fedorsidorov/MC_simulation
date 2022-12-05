import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import importlib
import indexes as ind
from functions import array_functions as af
from functions import MC_functions as mcf
from mapping import mapping_3um_500nm as mm
from functions import diffusion_functions
import constants as const

af = importlib.reload(af)
df = importlib.reload(diffusion_functions)
mm = importlib.reload(mm)
mcf = importlib.reload(mcf)
ind = importlib.reload(ind)
const = importlib.reload(const)

font = {'size': 14}
matplotlib.rc('font', **font)


# %% get Karlsson D
def get_Karlsson_D(T_C, wp):  # in cm^2 / s

    T_g = 119
    delta_T = T_C - T_g

    if wp < 0.795:
        coefs = -4.428, 1.842, 0, 8.12e-3
    elif wp < 0.927:
        coefs = 26.0, 37.0, 0.0797, 0
    elif wp < 0.945:
        # coefs = 159.0, 170.0, 0.3664, 0
        coefs = 15.9, 17.0, 0.3664, 0
    else:
        coefs = -13.7, 0.5, 0, 0

    C1, C2, C3, C4 = coefs
    log_D = wp * delta_T * C4 + (C1 - wp * C2) + delta_T * C3

    return 10**log_D


WP = np.linspace(0.8, 1, 101)

# T = 150
T = 119
DD = np.zeros(len(WP))

for i, wp in enumerate(WP):
    # print(wp)
    DD[i] = get_Karlsson_D(T, wp)


# plt.figure(dpi=300)
# plt.plot(WP, np.log10(DD))
# plt.ylim(-16, 0)
# plt.grid()
# plt.show()


# %%
def get_D_Mn_135(Mn):
    return 2e-10 * 10**(1.06e+4*(1/Mn - 1/30000))


def get_D_Mn_150(Mn):
    return 3.5e-10 * 10**(1.06e+4*(1/Mn - 1/30000))


# %%
MN = np.logspace(3, 6, 100)
DD = get_D_Mn_135(MN)

plt.figure(dpi=600, figsize=[4, 3])
# plt.figure(dpi=600, figsize=[3.4, 2.55])
plt.loglog(MN, DD)

plt.xlim(1e+3, 1e+6)
plt.ylim(1e-11, 1e+1)

plt.xlabel(r'$M_n$, г/моль')
plt.ylabel(r'$D$, см$^2$/c')
plt.grid()

plt.savefig('DD.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('DD_new.jpg', dpi=600, bbox_inches='tight')
plt.show()

# %%
# PMMA 950K
PD = 2.47
x_0 = 2714
z_0 = (2 - PD)/(PD - 1)
y_0 = x_0 / (z_0 + 1)

tau = np.load('notebooks/Boyd_kinetic_curves/arrays/tau.npy')
Mn_130 = np.load('notebooks/Boyd_kinetic_curves/arrays/Mn_130_term.npy') * 100

scission_weight = 0.08  # 130 C - 0.082748

xx_centers, zz_centers = mm.x_centers_50nm, mm.z_centers_50nm
xx_bins, zz_bins = mm.x_bins_50nm, mm.z_bins_50nm

n_e_DATA_files = 600
n_files_required_s = 60

time_step = 1

bin_volume = 50 * 100 * 50
bin_n_monomers = bin_volume / const.V_mon_nm3

scission_matrix = np.zeros((len(xx_centers), len(zz_centers)))
tau_matrix = np.zeros((len(xx_centers), len(zz_centers)))
Mn_matrix = np.ones((len(xx_centers), len(zz_centers))) * Mn_130[0]


# for time_cnt in range(1):
# for time_cnt in range(5):
for time_cnt in range(10):
# for time_cnt in range(50):

    print(time_cnt)

    now_scission_matrix = np.zeros((len(xx_centers), len(zz_centers)))

    for n in range(n_files_required_s):

        n_file = np.random.choice(n_e_DATA_files)

        now_x0 = np.random.normal(loc=0, scale=300)
        now_e_DATA_Pv = np.load('/Volumes/Transcend/e_DATA_500nm_point/0/e_DATA_Pv_' + str(n_file) + '.npy')

        scission_inds = np.where(np.random.random(len(now_e_DATA_Pv)) < scission_weight)[0]
        now_e_DATA_sci = now_e_DATA_Pv[scission_inds, :]

        now_e_DATA_sci[:, ind.e_DATA_x_ind] += now_x0

        af.snake_array(
            array=now_e_DATA_sci,
            x_ind=ind.e_DATA_x_ind,
            y_ind=ind.e_DATA_y_ind,
            z_ind=ind.e_DATA_z_ind,
            xyz_min=[mm.x_min, mm.y_min, -np.inf],
            xyz_max=[mm.x_max, mm.y_max, np.inf]
        )

        now_scission_matrix += np.histogramdd(
            sample=now_e_DATA_sci[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
            bins=[xx_bins, zz_bins]
        )[0]

        for i in range(len(xx_centers)):
            for j in range(len(zz_centers)):
                now_k_s = now_scission_matrix[i, j] / time_step / bin_n_monomers
                tau_matrix[i, j] += y_0 * now_k_s * time_step

        scission_matrix += now_scission_matrix


#%%
for i in range(len(xx_centers)):
    for j in range(len(zz_centers)):
        if tau_matrix[i, j] > tau[-1]:
            Mn_matrix[i, j] = Mn_130[-1]
        else:
            Mn_matrix[i, j] = mcf.lin_log_interp(tau, Mn_130)(tau_matrix[i, j])

# %%
# scission_matrix = np.load('notebooks/diffusion/scission_matrix_5s.npy')
# scission_matrix = np.load('notebooks/diffusion/scission_matrix_50s.npy')

# Mn_matrix = np.load('notebooks/diffusion/Mn_matrix_5s.npy')
Mn_matrix = np.load('notebooks/diffusion/Mn_matrix_50s.npy')

plt.figure(dpi=600)
# fig, ax = plt.subplots(dpi=600, figsize=[4, 8])
fig, ax = plt.subplots(dpi=600, figsize=[8, 6])

# plt.imshow(scission_matrix.transpose() / (0.05**2 * 0.1), extent=[-1500, 1500, 0, 500])
plt.imshow(Mn_matrix.transpose(), extent=[-1500, 1500, 0, 500])

# plt.title(r'$n_{sci}$ при $t$ = 5 c, 1/мкм$^3$', fontsize=14)
# plt.title(r'$n_{sci}$ при $t$ = 10 c, 1/мкм$^3$', fontsize=14)
# plt.title(r'$n_{sci}$ при $t$ = 50 c, 1/мкм$^3$', fontsize=14)

# plt.title(r'$M_n$ при $t$ = 5 c, г/моль', fontsize=14)
plt.title(r'$M_n$ при $t$ = 50 c, г/моль', fontsize=14)

plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

cbar = plt.colorbar(orientation='horizontal')
cbar.formatter.set_powerlimits((0, 0))
cbar.formatter.set_useMathText(True)

# plt.savefig('sci_conc_5s_20_10.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('sci_conc_5s_new.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('sci_conc_10s.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('sci_conc_50s_10_20.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('sci_conc_50s.jpg', dpi=600, bbox_inches='tight')

# plt.savefig('Mn_hist_5s_8_6.jpg', dpi=600, bbox_inches='tight')
# plt.savefig('Mn_hist_10s_8_6.jpg', dpi=600, bbox_inches='tight')
plt.savefig('Mn_hist_50s_8_6.jpg', dpi=600, bbox_inches='tight')

plt.show()

# %%
Mn_matrix = np.load('notebooks/diffusion/Mn_matrix_5s.npy')

# D_matrix = np.zeros(np.shape(Mn_matrix))
D_matrix = get_D_Mn_135(Mn_matrix)

plt.figure(dpi=600, figsize=[4, 3])

plt.imshow(D_matrix.transpose(), extent=[-1500, 1500, 0, 500])

plt.title(r'M$_n$ при t = 5 c, г/моль')
plt.xlabel(r'x, нм')
plt.ylabel(r'z, нм')
plt.colorbar(format='%.0e')

plt.show()
# plt.savefig('Mn_hist_5s.jpg', dpi=600, bbox_inches='tight')


# %%
# T = 130
scission_matrix = np.load('notebooks/diffusion/scission_matrix_1s.npy')
zip_length = 500
monomer_matrix = scission_matrix * zip_length

CC_0 = monomer_matrix / (mm.step_50nm * mm.ly * mm.step_50nm * 1e-21)

plt.figure(dpi=600, figsize=[4, 3])
plt.imshow(CC_0.transpose(), extent=[-1500, 1500, 0, 500])

plt.title('$n_{mon}$ при $t$ = 0 с, 1/см$^3$')
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

cbar = plt.colorbar()
cbar.formatter.set_powerlimits((0, 0))
cbar.formatter.set_useMathText(True)
plt.clim(0, 3e+22)

plt.show()
# plt.savefig('n_mon_hist_initial.jpg', dpi=600, bbox_inches='tight')

# %%
# D = df.get_D(130, 1)  # 2.4e-9
# D = 9.7e-11
D = 1e-9

xx = xx_centers * 1e-7
yy = zz_centers * 1e-7
h = mm.step_50nm * 1e-7

CC = deepcopy(CC_0)

Nx = len(xx)
Ny = len(yy)

t_end = 20
# t_end = 20
# t_end = 2
tau = t_end / 50

alphas_x = np.zeros(Nx)
betas_x = np.zeros(Nx)

alphas_y = np.zeros(Ny)
betas_y = np.zeros(Ny)

t = 0

while t < t_end:
    t += tau

    # Ox
    for j in range(Ny):

        # II
        alphas_x[0] = 2 * D * tau / (h ** 2 + 2 * D * tau)
        betas_x[0] = h ** 2 * CC[0, j] / (h ** 2 + 2 * D * tau)

        for i in range(1, Nx - 1):
            Ai = Ci = D / h**2
            Bi = 2 * D / h**2 + 1 / tau
            Fi = -CC[i, j] / tau

            alphas_x[i] = Ai / (Bi - Ci * alphas_x[i - 1])
            betas_x[i] = (Ci * betas_x[i - 1] - Fi) / (Bi - Ci * alphas_x[i - 1])

        # II
        CC[Nx - 1, j] = (2 * D * tau * betas_x[Nx - 2] + h ** 2 * CC[Nx - 1, j]) / \
                        (h ** 2 + 2 * D * tau * (1 - alphas_x[Nx - 2]))

        for i in range(Nx - 2, -1, -1):
            CC[i, j] = alphas_x[i] * CC[i + 1, j] + betas_x[i]

    # Oy
    for i in range(Nx):

        # I
        alphas_y[0] = 0
        betas_y[0] = 0  # c at the interface

        for j in range(1, Ny - 1):
            Ai = Ci = D / h ** 2
            Bi = 2 * D / h ** 2 + 1 / tau
            Fi = fi = -CC[i, j] / tau

            alphas_y[j] = Ai / (Bi - Ci * alphas_y[j - 1])
            betas_y[j] = (Ci * betas_y[j - 1] - Fi) / (Bi - Ci * alphas_y[j - 1])

        # II
        CC[i, Ny - 1] = (2 * D * tau * betas_y[Ny - 2] + h**2 * CC[i, Ny - 1]) /\
                        (h**2 + 2 * D * tau * (1 - alphas_y[Ny - 2]))

        for j in range(Ny - 2, -1, -1):
            CC[i, j] = alphas_y[j] * CC[i, j + 1] + betas_y[j]


plt.figure(dpi=600, figsize=[4, 3])
plt.imshow(CC.transpose(), extent=[-1500, 1500, 0, 500])

# plt.title('$n_{mon}^{' + str(t_end) + 'c}$,1/см$^3$ ($D$=9.7$\cdot$10$^{-11}$см$^2$/c)')
plt.title('$n_{mon}^{' + str(t_end) + 'c}$,1/см$^3$ ($D$=1$\cdot$10$^{-9}$см$^2$/c)')
# plt.title('$n_{mon}$ при $t$ = 2 с, 1/см$^3$')
plt.xlabel(r'$x$, нм')
plt.ylabel(r'$z$, нм')

cbar = plt.colorbar()
cbar.formatter.set_powerlimits((0, 0))
cbar.formatter.set_useMathText(True)
# plt.clim(0, 1.2e+21)
# plt.clim(0, 4.5e+21)

# plt.show()
# plt.savefig('n_mon_hist_9p7e-11_' + str(t_end) + 's.jpg', dpi=600, bbox_inches='tight')
plt.savefig('n_mon_hist_1e-9_' + str(t_end) + 's.jpg', dpi=600, bbox_inches='tight')



