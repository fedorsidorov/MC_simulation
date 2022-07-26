import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
import grid
from scipy import interpolate
import constants as const
import indexes
from functions import MC_functions as mcf

indexes = importlib.reload(indexes)
const = importlib.reload(const)
mcf = importlib.reload(mcf)


#%%
def interpolate_diff_cs(EE_raw, THETA_deg_raw, diff_cs_raw, extrap):

    EE_raw_ext = np.concatenate((grid.EE[:1], EE_raw, grid.EE[-1:]))
    diff_cs_ext = np.concatenate((diff_cs_raw[:1, :],
                                  diff_cs_raw, diff_cs_raw[-1:, :]), axis=0)

    if extrap:
        diff_cs_ext[0, :] = np.exp(
            np.log(EE_raw_ext[0]/EE_raw_ext[1]) /
            np.log(EE_raw_ext[1]/EE_raw_ext[2]) *
            np.log(diff_cs_ext[1, :]/diff_cs_ext[2, :])
            ) * diff_cs_ext[1, :]
        
        diff_cs_ext[-1, :] = np.exp(
            np.log(EE_raw_ext[-1]/EE_raw_ext[-2]) /
            np.log(EE_raw_ext[-2]/EE_raw_ext[-3]) *
            np.log(diff_cs_ext[-2]/diff_cs_ext[-3, :])
            ) * diff_cs_ext[-2, :]

    diff_cs_pre =\
        np.power(10, interpolate.interp1d(THETA_deg_raw, np.log10(diff_cs_ext), axis=1)(grid.THETA_deg))
    diff_cs = np.power(10, interpolate.interp1d(EE_raw_ext, np.log10(diff_cs_pre), axis=0)(grid.EE))

    return diff_cs


def interpolate_cs(EE_raw, cs_raw, extrap):
    
    EE_raw_ext = np.concatenate((grid.EE[:1], EE_raw, grid.EE[-1:]))
    cs_ext = np.concatenate((cs_raw[:1], cs_raw, cs_raw[-1:]), axis=0)

    if extrap:
        cs_ext[0] = np.exp(
            np.log(EE_raw_ext[0]/EE_raw_ext[1]) /
            np.log(EE_raw_ext[1]/EE_raw_ext[2]) *
            np.log(cs_ext[1]/cs_ext[2])
            ) * cs_ext[1]
        
        cs_ext[-1] = np.exp(
            np.log(EE_raw_ext[-1]/EE_raw_ext[-2]) /
            np.log(EE_raw_ext[-2]/EE_raw_ext[-3]) *
            np.log(cs_ext[-2]/cs_ext[-3])
            ) * cs_ext[-2]

    cs = mcf.log_log_interp(EE_raw_ext, cs_ext)(grid.EE)
    
    return cs


#%%
EE_raw = np.load('notebooks/elastic/raw_arrays/elsepa_EE.npy')
THETA_deg_raw = np.load('notebooks/elastic/raw_arrays/elsepa_theta.npy')


# %%
for model in ['easy', 'atomic', 'muffin']:
    for element in ['H', 'C', 'O', 'Si']:
        
        print(element)
        
        diff_cs_raw = np.load(os.path.join(
            'notebooks/elastic/raw_arrays', model, element, element + '_' + model + '_diff_cs.npy'
            ))
        
        cs_raw = np.load(os.path.join(
            'notebooks/elastic/raw_arrays', model, element, element + '_' + model + '_cs.npy'
            ))
        
        diff_cs = interpolate_diff_cs(EE_raw, THETA_deg_raw, diff_cs_raw, extrap=False)
        diff_cs_extrap = interpolate_diff_cs(EE_raw, THETA_deg_raw, diff_cs_raw, extrap=True)

        cs = interpolate_cs(EE_raw, cs_raw, extrap=False)
        cs_extrap = interpolate_cs(EE_raw, cs_raw, extrap=True)
        
        np.save(os.path.join('notebooks/elastic/final_arrays', model, element,
                             element + '_' + model + '_diff_cs.npy'), diff_cs)
        np.save(os.path.join('notebooks/elastic/final_arrays', model, element,
                             element + '_' + model + '_diff_cs_extrap.npy'), diff_cs_extrap)

        np.save(os.path.join('notebooks/elastic/final_arrays', model, element,
                             element + '_' + model + '_cs.npy'), cs)
        np.save(os.path.join('notebooks/elastic/final_arrays', model, element,
                             element + '_' + model + '_cs_extrap.npy'), cs_extrap)

# %% check cs interpolation
element = 'Si'

raw_cs = np.load('notebooks/elastic/raw_arrays/easy/' + element + '/' + element + '_easy_cs.npy')
final_cs = np.load('notebooks/elastic/final_arrays/easy/'
                        + element + '/' + element + '_easy_cs_extrap.npy')

plt.figure(dpi=300)
plt.loglog(EE_raw, raw_cs)
plt.loglog(grid.EE, final_cs, '--')
plt.show()

# %% check diff cs interpolation
element = 'Si'

diff_cs_raw = np.load('notebooks/elastic/raw_arrays/easy/' + element + '/' + element + '_easy_diff_cs.npy')
final_diff_cs = np.load('notebooks/elastic/final_arrays/easy/'
                        + element + '/' + element + '_easy_diff_cs_extrap.npy')

EE_raw = np.load('notebooks/elastic/raw_arrays/elsepa_EE.npy')
THETA_deg_raw = np.load('notebooks/elastic/raw_arrays/elsepa_theta.npy')
# diff_cs_raw = np.load('notebooks/elastic/raw_arrays/easy/H/H_easy_diff_cs.npy')

ans = interpolate_diff_cs(EE_raw, THETA_deg_raw, diff_cs_raw, extrap=False)

plt.figure(dpi=300)

plt.semilogy(THETA_deg_raw, diff_cs_raw[21], '-')
plt.semilogy(grid.THETA_deg, ans[indexes.E_100], '--')

plt.semilogy(THETA_deg_raw, diff_cs_raw[34], '-')
plt.semilogy(grid.THETA_deg, ans[indexes.E_1000], '--')

plt.semilogy(THETA_deg_raw, diff_cs_raw[42], '-')
plt.semilogy(grid.THETA_deg, ans[indexes.E_5000], '--')

plt.semilogy(THETA_deg_raw, diff_cs_raw[47], '-')
plt.semilogy(grid.THETA_deg, ans[indexes.E_10000], '--')

plt.semilogy(THETA_deg_raw, diff_cs_raw[57], '-')
plt.semilogy(grid.THETA_deg, ans[indexes.E_20000], '--')

plt.grid()
plt.show()


# %%
MELEC = '4'
MEXCH = '1'
# MCPOL = '0'

# for MELEC in ['1', '2', '3', '4']:
# for MEXCH in ['0', '1', '2', '3']:
for MCPOL in ['0', '1', '2']:

    diff_cs_raw = np.load(
        'notebooks/elastic/raw_arrays/root_Hg/root_' + MELEC + MEXCH + MCPOL + '_diff_cs.npy'
        # 'notebooks/elastic/raw_arrays/root_Si/root_' + MELEC + MEXCH + MCPOL + '_diff_cs.npy'
    )

    cs_raw = np.load(
        'notebooks/elastic/raw_arrays/root_Hg/root_' + MELEC + MEXCH + MCPOL + '_cs.npy'
        # 'notebooks/elastic/raw_arrays/root_Si/root_' + MELEC + MEXCH + MCPOL + '_cs.npy'
    )

    diff_cs = interpolate_diff_cs(EE_raw, THETA_deg_raw, diff_cs_raw, extrap=False)
    cs = interpolate_cs(EE_raw, cs_raw, extrap=False)

    np.save('notebooks/elastic/final_arrays/root_Hg/root_' + MELEC + MEXCH + MCPOL + '_diff_cs.npy', diff_cs)
    np.save('notebooks/elastic/final_arrays/root_Hg/root_' + MELEC + MEXCH + MCPOL + '_cs.npy', cs)

    # np.save('notebooks/elastic/final_arrays/root_Si/root_' + MELEC + MEXCH + MCPOL + '_diff_cs.npy', diff_cs)
    # np.save('notebooks/elastic/final_arrays/root_Si/root_' + MELEC + MEXCH + MCPOL + '_cs.npy', cs)







