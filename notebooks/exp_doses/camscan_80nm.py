import numpy as np

# %%
pitch = 3.3e-4  # cm

j_exp_area = 1.9e-9  # A / cm
# j_exp_line = 1.9e-9 * pitch

dose_area = 0.6e-6  # C / cm^2
dose_line = dose_area * pitch

t = dose_area / j_exp_area  # 316 s

dt = 1  # s
Q = dose_s * area
n_electrons = Q / constants.e_SI  # 2 472
n_electrons_s = int(np.around(n_electrons / t))
