import importlib
import numpy as np
import matplotlib.pyplot as plt
import os
from functions import chain_functions as cf
from functions import array_functions as af
from tqdm import tqdm
import constants_mapping as const_m
import constants_physics as const_p

cf = importlib.reload(cf)
af = importlib.reload(af)
const_m = importlib.reload(const_m)
const_p = importlib.reload(const_p)

# %%
source_dir = '/Volumes/ELEMENTS/Chains/'
print(os.listdir(source_dir))

# %%
chains = []

for folder in os.listdir(source_dir):
    if '.' in folder:
        continue

    print(folder)

    for i, fname in enumerate(os.listdir(os.path.join(source_dir, folder))):
        if 'DS_Store' in fname or '._' in fname:
            continue
        chains.append(np.load(os.path.join(source_dir, folder, fname)))

# %% create chain_list and check density
mass_array = np.load('Resources/Harris/harris_x_before.npy')
molecular_weight_array = np.load('Resources/Harris/harris_y_before_fit.npy')
# plt.semilogx(mass_array, molecular_weight_array, 'ro')
# plt.show()

# %%
V = np.prod(const_m.l_xyz) * 1e-7 ** 3  # cm^3
n_monomers_required = const_p.rho_PMMA * V / const_p.m_MMA

# %%
hist_total = np.zeros((1, 1, 1))
hist_10nm = np.zeros((len(const_m.x_bins_10nm) - 1, len(const_m.y_bins_10nm) - 1, len(const_m.z_bins_10nm) - 1))
hist_2nm = np.zeros((len(const_m.x_bins_2nm) - 1, len(const_m.y_bins_2nm) - 1, len(const_m.z_bins_2nm) - 1))

chain_list = []
chain_lens = []

i = 0

n_mon_now = 0
n_mon_previous = 0

progress_bar = tqdm(total=n_monomers_required, position=0)

while True:
    if n_mon_previous > n_monomers_required:
        print('Needed density is achieved')
        break

    now_chain_base = np.random.choice(chains)

    now_len = cf.get_chain_len(mass_array, molecular_weight_array)
    chain_lens.append(now_len)

    beg_ind = np.random.randint(len(now_chain_base) - now_len)
    now_chain = now_chain_base[beg_ind:beg_ind + now_len]

    shift = np.random.uniform(const_p.xyz_min, const_p.xyz_max)
    now_chain_shifted = now_chain + shift

    af.snake_array(now_chain_shifted, 0, 1, 2, const_m.xyz_min, const_m.xyz_max)

    if np.any(np.min(now_chain_shifted, axis=0) < const_m.xyz_min) or \
            np.any(np.max(now_chain_shifted, axis=0) >= const_m.xyz_max):
        print('chain snaking error, i =', i)

    hist_total += np.histogramdd(now_chain_shifted, bins=const_m.bins_total)[0]
    hist_10nm += np.histogramdd(now_chain_shifted, bins=const_m.bins_10nm)[0]
    hist_2nm += np.histogramdd(now_chain_shifted, bins=const_m.bins_2nm)[0]

    n_monomers_now = np.sum(hist_total)

    if n_monomers_now > n_mon_previous:
        chain_list.append(now_chain_shifted)
    else:
        print('n monomers doesn\'t increase!')

    n_mon_previous = n_mon_now

    progress_bar.update(1)

    i += 1

# %%
chain_lens = np.zeros(len(chain_list))

for i in range(len(chain_list)):
    chain_lens[i] = len(chain_list[i])

# %%
print('2nm average =', np.average(hist_2nm))
print('2nm average density =', np.average(hist_2nm) * m_mon / (2e-7) ** 3)

# %%
total_rho = np.sum(chain_lens) * m_mon / V

print('total rho =', total_rho)

# %%
n_empty = 50 * 50 * 250 - np.count_nonzero(hist_2nm)
part_empty = n_empty / (50 * 50 * 250)

print(part_empty)

# %% save chains to files
source_dir = os.path.join(mc.sim_folder, 'Chains_Harris_2020')

i = 0

for chain in chain_list:
    mu.pbar(i, len(chain_list))
    np.save(os.path.join(source_dir, 'chain_shift_' + str(i) + '.npy'), chain)
    i += 1

# %%
np.save(os.path.join(mc.sim_folder, 'Chains_Harris_2020', 'hist_2nm.npy'), hist_2nm)

# %%
source_dir = os.path.join(mc.sim_folder, 'Chains', 'Chains_Harris_2020')

lens = []

files = os.listdir(source_dir)

for file in files:

    if 'DS' in file or 'hist' in file:
        continue

    chain = np.load(os.path.join(source_dir, file))

    lens.append(len(chain))

chain_lens = np.array(lens)

# %%
distr = np.load('harris_initial_distr.npy')

m = distr[:, 0]
mw = distr[:, 1]

mass = np.array(chain_lens) * 100

bins = np.logspace(2, 7.1, 21)

plt.hist(mass, bins, label='sample')
plt.gca().set_xscale('log')

plt.plot(m, mw * 0.6e+5, label='harris')

plt.title('Harris chain sample, NO period, 100 nm offser')
plt.xlabel('molecular weight')
plt.ylabel('density')

plt.xlim(1e+3, 1e+8)

plt.legend()
plt.grid()
plt.show()

# plt.savefig('Harris_sample_2020.png', dpi=300)


# %% check density
density_total = hist_total[0][0][0] * m_mon / V
# density_prec = hist_prec * m_mon / V * (len(x_bins_prec) - 1) * (len(y_bins_prec) - 1) *\
#    (len(z_bins_prec) - 1)


# %%
n_mon_max = hist_2nm.max()
print(np.sum(hist_2nm) * m_mon / V)

# %% cut chains to cube shape
# chain_cut_list = []
#
# for chain in chain_list:
#
#    statements = [chain[:, 0] >= x_min, chain[:, 0] <= x_max,
#                  chain[:, 1] >= y_min, chain[:, 1] <= y_max]
#    inds = np.where(np.logical_and.reduce(statements))[0]
#
#    beg = 0
#    end = -1
#
#    for i in range(len(inds) - 1):
#        if inds[i+1] > inds[i] + 1 or i == len(inds) - 2:
#            end = i + 1
#            chain_cut_list.append(chain[inds[beg:end], :])
#            beg = i + 1


# %% get nice 3D picture
# l_x, l_y, l_z = l_xyz
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for chain in chain_list[0:-1:50]:
#    ax.plot(chain[:, 0], chain[:, 1], chain[:, 2])
#
# ax.plot(np.linspace(x_min, x_max, l_x), np.ones(l_x)*y_min, np.ones(l_x)*z_min, 'k')
# ax.plot(np.linspace(x_min, x_max, l_x), np.ones(l_x)*y_max, np.ones(l_x)*z_min, 'k')
# ax.plot(np.linspace(x_min, x_max, l_x), np.ones(l_x)*y_min, np.ones(l_x)*z_max, 'k')
# ax.plot(np.linspace(x_min, x_max, l_x), np.ones(l_x)*y_max, np.ones(l_x)*z_max, 'k')
#
# ax.plot(np.ones(l_y)*x_min, np.linspace(y_min, y_max, l_y), np.ones(l_y)*z_min, 'k')
# ax.plot(np.ones(l_y)*x_max, np.linspace(y_min, y_max, l_y), np.ones(l_y)*z_min, 'k')
# ax.plot(np.ones(l_y)*x_min, np.linspace(y_min, y_max, l_y), np.ones(l_y)*z_max, 'k')
# ax.plot(np.ones(l_y)*x_max, np.linspace(y_min, y_max, l_y), np.ones(l_y)*z_max, 'k')
#
# ax.plot(np.ones(l_z)*x_min, np.ones(l_z)*y_min, np.linspace(z_min, z_max, l_z), 'k')
# ax.plot(np.ones(l_z)*x_max, np.ones(l_z)*y_min, np.linspace(z_min, z_max, l_z), 'k')
# ax.plot(np.ones(l_z)*x_min, np.ones(l_z)*y_max, np.linspace(z_min, z_max, l_z), 'k')
# ax.plot(np.ones(l_z)*x_max, np.ones(l_z)*y_max, np.linspace(z_min, z_max, l_z), 'k')
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.title('Polymer chain simulation')
# ax.set_xlabel('x, nm')
# ax.set_ylabel('y, nm')
# ax.set_zlabel('z, nm')
# plt.show()

