import importlib
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from mapping import mapping_5um_900nm as mm
from functions import array_functions as af
from functions import MC_functions as mcf
from functions import e_matrix_functions as emf
from functions import scission_functions as sf
from functions import reflow_functions as rf
from functions import SE_functions as ef
from functions import e_beam_MC as eb_MC

import indexes as ind

const = importlib.reload(const)
eb_MC = importlib.reload(eb_MC)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
mcf = importlib.reload(mcf)
mm = importlib.reload(mm)
af = importlib.reload(af)
ef = importlib.reload(ef)
sf = importlib.reload(sf)
rf = importlib.reload(rf)

# %%
j_exp_s = 0.85e-9  # A / cm^2 --- exposure current
j_exp_l = j_exp_s * mm.lx_cm  # A / cm --- current per Y unit length

# dose_s = 0.05e-6  # C / cm^2 --- dose per area
dose_s = 0.2e-6  # C / cm^2 --- dose per area
# dose_s = 0.87e-6  # C / cm^2

dose_l = dose_s * mm.lx_cm  # C / cm --- dose per Y unit length
total_time = dose_l / j_exp_l  # 235 s
Q = dose_s * mm.area_cm2
n_electrons = Q / constants.e_SI  # 12 483
n_electrons_s = int(np.around(n_electrons / total_time))

step_time = 1
primary_electrons_in_file = n_electrons_s * step_time

n_steps = int(total_time / step_time)

r_beam = 100
scission_weight = 0.092  # 160 C

r_beam_x = 100
r_beam_y = mm.ly / 2

E0 = 20e+3

# %%
profile_0p2 = np.loadtxt('notebooks/DEBER_simulation/exp_curves/exp_900nm_5um_0.2uC_cm2.txt')

profile_0p2[0, 1] = profile_0p2[:, 1].max()
profile_0p2[-1, 1] = profile_0p2[:, 1].max()

xx_vac = profile_0p2[:, 0] * 1000 - 6935.08
zz_vac = (profile_0p2[:, 1].max() - profile_0p2[:, 1]) * 1000

plt.figure(dpi=300)

plt.plot(xx_vac, zz_vac)
plt.plot(xx_vac, zz_vac * 0.7)
plt.plot(xx_vac, zz_vac * 0.5)
plt.plot(xx_vac, zz_vac * 0.3)
plt.grid()
plt.show()

# %% Area
area = np.trapz(zz_vac, x=xx_vac)

# %%
zz_vac_list = []
total_tau_array = np.zeros(len(mm.x_centers_20nm))

plt.figure(dpi=300)

for n_step in range(n_steps):

    zz_vac = zz_vac * n_step / n_steps
    print(n_step, 'zz_vac_max =', np.max(zz_vac))

    xx_vac_for_sim = np.concatenate(([-1e+6], xx_vac, [1e+6]))
    zz_vac_for_sim = np.concatenate(([zz_vac[0]], zz_vac, [zz_vac[-1]]))

    now_e_DATA = eb_MC.track_all_electrons(
        # n_electrons=int(primary_electrons_in_file / 2),
        n_electrons=primary_electrons_in_file,
        E0=E0,
        d_PMMA=mm.d_PMMA,
        z_cut=np.inf,
        Pn=True,
        xx_vac=xx_vac_for_sim,
        zz_vac=zz_vac_for_sim,
        r_beam_x=r_beam_x,
        r_beam_y=r_beam_y
    )

    now_Pv_e_DATA = now_e_DATA[np.where(
        np.logical_and(
            now_e_DATA[:, ind.e_DATA_layer_id_ind] == ind.PMMA_ind,
            now_e_DATA[:, ind.e_DATA_process_id_ind] == ind.sim_PMMA_ee_val_ind))
    ]

    af.snake_array(
            array=now_Pv_e_DATA,
            x_ind=ind.e_DATA_x_ind,
            y_ind=ind.e_DATA_y_ind,
            z_ind=ind.e_DATA_z_ind,
            xyz_min=[mm.x_min, mm.y_min, -np.inf],
            xyz_max=[mm.x_max, mm.y_max, np.inf]
        )

    now_Pv_e_DATA = emf.delete_snaked_vacuum_events(now_Pv_e_DATA, xx_vac_for_sim, zz_vac_for_sim)

    val_matrix = np.histogramdd(
        sample=now_Pv_e_DATA[:, [ind.e_DATA_x_ind, ind.e_DATA_z_ind]],
        bins=[mm.x_bins_5nm, mm.z_bins_5nm]
    )[0]

    # val_matrix += val_matrix[::-1, :]

    scission_matrix = np.zeros(np.shape(val_matrix), dtype=int)

    for x_ind in range(len(val_matrix)):
        for z_ind in range(len(val_matrix[0])):
            n_val = int(val_matrix[x_ind, z_ind])

            scissions = np.where(np.random.random(n_val) < scission_weight)[0]
            scission_matrix[x_ind, z_ind] = len(scissions)

    np.save('notebooks/DEBER_simulation/vary_zz_vac_0p2_1s/zz_vac_' + str(n_step) + '_check.npy', zz_vac)
    np.save('notebooks/DEBER_simulation/vary_zz_vac_0p2_1s/scission_matrix_' + str(n_step) + '_check.npy', scission_matrix)
