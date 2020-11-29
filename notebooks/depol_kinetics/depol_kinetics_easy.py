import numpy as np
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from mapping import mapping_viscosity_80nm as mm
from functions import DEBER_functions as deber
from functions import mapping_functions as mf
from functions import e_matrix_functions as emf
from functions import array_functions as af
from functions import scission_functions as sf
import constants as const
import indexes as ind

mapping = importlib.reload(mm)
deber = importlib.reload(deber)
const = importlib.reload(const)
emf = importlib.reload(emf)
ind = importlib.reload(ind)
af = importlib.reload(af)
sf = importlib.reload(sf)
mf = importlib.reload(mf)

# %% j = 80e-9  # A / cm^2
xx = mm.x_centers_10nm
zz_vac = np.zeros(len(xx))

source = 'data/e_DATA_Pv_80nm/'
n_files_total = 500

E0 = 20e+3

D = 20e-6  # C / cm^2
total_time = 250  # s
Q = D * mm.area_cm2

# T = 125 C
weight = 0.275

n_electrons_required = Q / const.e_SI
n_files_required = int(n_electrons_required / 100)

n_primaries_in_file = 100

# %%
now_z_vac = 0
now_z_vac_list = []

n_radicals = 0
n_monomers = 0

t_step = 1e-6  # s
# n_steps = int(total_time / t_step)
n_steps = 100000

p_depol = 0.1
p_term_1 = 0.1
p_nothing = 1 - p_depol - p_term_1

inds = [0, 1, 2]
probs = [p_depol, p_term_1, p_nothing]

# progress_bar = tqdm(total=n_steps, position=0)

for i in range(n_steps):

    print(i)

    if i % 5000 == 0:

        now_z_vac_list.append(now_z_vac)

        e_DATA, e_DATA_PMMA_val = deber.get_e_DATA_PMMA_val(
            xx=xx,
            zz_vac=np.zeros(len(xx))*now_z_vac,
            d_PMMA=mm.d_PMMA,
            n_electrons=1,
            E0=E0,
            r_beam_x=mm.lx / 2,
            r_beam_y=mm.ly / 2
        )
        scission_matrix, e_matrix_E_dep = deber.get_scission_matrix(e_DATA_PMMA_val, weight=weight)

        n_radicals += int(np.sum(scission_matrix))

    for _ in range(n_radicals):

        ind = np.random.choice(inds, p=probs)

        if ind == 0:
            n_monomers += 1
        elif ind == 1:
            n_radicals -= 1

    V_monomers = n_monomers * const.V_mon_nm3
    dh_monomers = V_monomers / mm.area_nm2

    now_z_vac += dh_monomers

    # progress_bar.update()


