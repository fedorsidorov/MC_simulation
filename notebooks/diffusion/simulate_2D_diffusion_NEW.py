import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import importlib
import indexes as ind
from functions import array_functions as af
from functions import MC_functions as mcf
from mapping import mapping_3um_500nm as mm
from functions import e_matrix_functions as emf
from functions import diffusion_functions
from functions import reflow_functions as rf
import constants as const
from matplotlib.ticker import LogFormatter
from matplotlib import colors

af = importlib.reload(af)
df = importlib.reload(diffusion_functions)
mm = importlib.reload(mm)
emf = importlib.reload(emf)
mcf = importlib.reload(mcf)
rf = importlib.reload(rf)
ind = importlib.reload(ind)
const = importlib.reload(const)

font = {'size': 14}
matplotlib.rc('font', **font)


# %%
scission_weight = 0.05  # room
# scission_weight = 0.08  # 130 C - 0.082748

xx_centers, zz_centers = mm.x_centers_50nm, mm.z_centers_50nm
xx_bins, zz_bins = mm.x_bins_50nm, mm.z_bins_50nm

# n_e_DATA_files = 600
n_e_DATA_files = 1000
n_files_required_s = 60

time_step = 1

bin_volume = 50 * 100 * 50
bin_n_monomers = bin_volume / const.V_mon_nm3

# %%
scission_matrix = np.zeros((len(xx_centers), len(zz_centers)))
val_matrix = np.zeros((len(xx_centers), len(zz_centers)))
E_dep_matrix = np.zeros((len(xx_centers), len(zz_centers)))

# %
# for time_cnt in range(1):
# for time_cnt in range(5):
# for time_cnt in range(10):
for time_cnt in range(34):
# for time_cnt in range(50):
# for time_cnt in range(200):

    print(time_cnt)

    for n in range(n_files_required_s):

        n_file = np.random.choice(n_e_DATA_files)

        now_e_DATA_Pn = np.load('/Volumes/Transcend/e_DATA_500nm_point/e_DATA_Pn_' + str(n_file) + '.npy')
        now_x0 = np.random.uniform(-150, 150)
        now_e_DATA_Pn[:, ind.e_DATA_x_ind] += now_x0

        emf.rotate_DATA(now_e_DATA_Pn)

        af.snake_array(
            array=now_e_DATA_Pn,
            x_ind=ind.e_DATA_x_ind,
            y_ind=ind.e_DATA_y_ind,
            z_ind=ind.e_DATA_z_ind,
            xyz_min=[-np.inf, mm.y_min, -np.inf],
            xyz_max=[np.inf, mm.y_max, np.inf]
        )

        now_e_DATA_E_dep = now_e_DATA_Pn[np.where(now_e_DATA_Pn[:, ind.e_DATA_E_dep_ind] > 0)]
        # now_e_DATA_E_dep = now_e_DATA_Pn[np.where(now_e_DATA_Pn[:, ind.e_DATA_E_dep_ind] < 1000)]

        E_dep_matrix += np.histogramdd(
            sample=now_e_DATA_E_dep[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
            bins=[xx_bins, zz_bins],
            weights=now_e_DATA_E_dep[:, ind.e_DATA_E_dep_ind]
        )[0]

        now_e_DATA_Pv = now_e_DATA_Pn[
            np.where(
                now_e_DATA_Pn[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind
            )
        ]

        val_matrix += np.histogramdd(
            sample=now_e_DATA_Pv[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
            bins=[xx_bins, zz_bins]
        )[0]

        scission_inds = np.where(np.random.random(len(now_e_DATA_Pv)) < scission_weight)[0]
        now_e_DATA_sci = now_e_DATA_Pv[scission_inds, :]

        scission_matrix += np.histogramdd(
            sample=now_e_DATA_sci[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
            bins=[xx_bins, zz_bins]
        )[0]


# %%
scission_matrix = scission_matrix[10:-10, :]
val_matrix = val_matrix[10:-10, :]
E_dep_matrix = E_dep_matrix[10:-10, :]

# %%
# scission_matrix = np.load('notebooks/diffusion/sci_mat_50s_2um_1.npy')

n_sci_matrix = scission_matrix.transpose() / (0.05**2 * 0.1)

plt.figure(dpi=600)
fig, ax = plt.subplots(dpi=600, figsize=[8, 6])

im = plt.imshow(
    n_sci_matrix,
    vmin=np.min(n_sci_matrix),
    vmax=np.max(n_sci_matrix),
    extent=[-1000, 1000, 0, 500],
    norm=colors.LogNorm()
)

plt.title(r'$n_{sci}$ (1/$\mu$m$^3$) at $D_l$ = 1 nC/cm', fontsize=14)
plt.xlabel(r'$x$, nm')
plt.ylabel(r'$z$, nm')

cbar = plt.colorbar(im, orientation='horizontal')

# plt.savefig('sci_conc_1uC_cm_LOG.jpg', dpi=600, bbox_inches='tight')

plt.show()


# %%
Mn_0 = 271400
Mn_matrix = np.ones(np.shape(scission_matrix)) * Mn_0

p_s_matrix = scission_matrix / bin_n_monomers
Mf_matrix_easy = 1 / (p_s_matrix / 100 + 1 / Mn_matrix)


plt.figure(dpi=600)
fig, ax = plt.subplots(dpi=600, figsize=[8, 6])

im = plt.imshow(
    Mf_matrix_easy.transpose(),
    vmin=np.min(Mf_matrix_easy),
    vmax=np.max(Mf_matrix_easy),
    extent=[-1000, 1000, 0, 500],
    norm=colors.LogNorm()
)

plt.title(r'$M_n^\prime$ (g/mol) at $D_l$ = 1 nC/cm', fontsize=14)
plt.xlabel(r'$x$, nm')
plt.ylabel(r'$z$, nm')

cbar = plt.colorbar(im, orientation='horizontal')

plt.savefig('Mf_mat_easy_1_nC_cm_LOG.jpg', dpi=600, bbox_inches='tight')

plt.show()

# %%
eta_matrix = np.zeros(np.shape(scission_matrix))

for i in range(len(eta_matrix)):
    for j in range(len(eta_matrix[0])):
        eta_matrix[i, j] = rf.get_viscosity_experiment_Mn(120, Mf_matrix_easy[i, j], 3.4, 1.4)

plt.figure(dpi=600)
fig, ax = plt.subplots(dpi=600, figsize=[8, 6])

im = plt.imshow(
    eta_matrix.transpose(),
    vmin=np.min(eta_matrix),
    vmax=np.max(eta_matrix),
    extent=[-1000, 1000, 0, 500],
    norm=colors.LogNorm()
)

plt.title(r'$\eta$ (Pa$\cdot$s) at $D_l$ = 1 nC/cm, $T$ = 120 °C', fontsize=14)
plt.xlabel(r'$x$, nm')
plt.ylabel(r'$z$, nm')

cbar = plt.colorbar(im, orientation='horizontal')

# plt.savefig('eta_hist_1_nC_cm_120C_LOG.jpg', dpi=600, bbox_inches='tight')

plt.show()

# %%
eta_array = np.average(eta_matrix, axis=1)

plt.figure(dpi=600)
fig, ax = plt.subplots(dpi=600, figsize=[4, 3])

plt.semilogy(xx_centers[10:-10], eta_array)

plt.title(r'averaged $\eta$ at $D_l$ = 1 nC/cm, $T$ = 120 °C', fontsize=14)
plt.xlabel(r'$x$, nm')
plt.ylabel(r'$\eta$')

plt.xlim(-1000, 1000)
# plt.ylim(1e-6, 1e-2)

plt.grid()
plt.savefig('eta_arr_1_nC_cm_120C_LOG.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %%
mob_matrix = np.zeros(np.shape(scission_matrix))

for i in range(len(eta_matrix)):
    for j in range(len(eta_matrix[0])):
        mob_matrix[i, j] = rf.get_SE_mobility(eta_matrix[i, j])

plt.figure(dpi=600)
fig, ax = plt.subplots(dpi=600, figsize=[8, 6])

im = plt.imshow(
    mob_matrix.transpose(),
    vmin=np.min(mob_matrix),
    vmax=np.max(mob_matrix),
    extent=[-1000, 1000, 0, 500],
    norm=colors.LogNorm()
)

plt.title(r'$\mu$ at $D_l$ = 1 nC/cm, $T$ = 120 °C', fontsize=14)

plt.xlabel(r'$x$, nm')
plt.ylabel(r'$z$, nm')

cbar = plt.colorbar(im, orientation='horizontal')

plt.savefig('mob_hist_1_nC_cm_120C_LOG.jpg', dpi=600, bbox_inches='tight')

plt.show()

# %%
mob_array = np.average(mob_matrix, axis=1)

plt.figure(dpi=600)
fig, ax = plt.subplots(dpi=600, figsize=[4, 3])

plt.semilogy(xx_centers[10:-10], mob_array)

plt.title(r'averaged $\mu$ at $D_l$ = 1 nC/cm, $T$ = 120 °C', fontsize=14)
plt.xlabel(r'$x$, nm')
plt.ylabel(r'$\mu$')

plt.xlim(-1000, 1000)
# plt.ylim(1e-6, 1e-2)

plt.grid()
plt.savefig('mob_arr_1_nC_cm_120C_LOG.jpg', dpi=600, bbox_inches='tight')
plt.show()


# %%
plt.figure(dpi=600)
fig, ax = plt.subplots(dpi=600, figsize=[8, 6])

ratio_matrix = scission_matrix / (1.6e-2 * E_dep_matrix)

plt.imshow(ratio_matrix.transpose(), extent=[-1000, 1000, 0, 500])

im = plt.imshow(
    ratio_matrix.transpose(),
    vmin=np.min(ratio_matrix),
    vmax=np.max(ratio_matrix),
    extent=[-1000, 1000, 0, 500],
    norm=colors.LogNorm()
)

plt.title(r'ratio_matrix', fontsize=14)

plt.xlabel(r'$x$, nm')
plt.ylabel(r'$z$, nm')

cbar = plt.colorbar(im, orientation='horizontal')

plt.savefig('ratio_1nC_cm.jpg', dpi=600, bbox_inches='tight')

plt.show()


# %% simulate room T Mn
# G_s = 1.9e-2
G_s = 1.6e-2

Mn_0 = 271400

Mn_matrix = np.ones(np.shape(scission_matrix)) * Mn_0
Mf_matrix = Mn_matrix / (1 + G_s * E_dep_matrix / (0.05**2 * 0.1 * 1e-12) * Mn_matrix / (const.rho_PMMA * const.Na))
# Mf_matrix = Mn_matrix / (1 + scission_matrix / (0.05**2 * 0.1 * 1e-12) * Mn_matrix / (const.rho_PMMA * const.Na))

plt.figure(dpi=600)
fig, ax = plt.subplots(dpi=600, figsize=[8, 6])

plt.imshow(Mf_matrix.transpose(), extent=[-1000, 1000, 0, 500])

im = plt.imshow(
    Mf_matrix.transpose(),
    vmin=np.min(Mf_matrix),
    vmax=np.max(Mf_matrix),
    extent=[-1000, 1000, 0, 500],
    norm=colors.LogNorm()
)

plt.title(r'$M_\mathrm{n}$ at $t$ = 20 s, E_DEP', fontsize=14)

plt.xlabel(r'$x$, nm')
plt.ylabel(r'$z$, nm')

cbar = plt.colorbar(im, orientation='horizontal')

plt.savefig('Mn_hist_E_DEP_1nC_cm.jpg', dpi=600, bbox_inches='tight')

plt.show()
