import numpy as np

# %% mapping Harris matrices
x_min, x_max = -50, 50
y_min, y_max = -50, 50
z_min, z_max = 0, 500

l_xyz = np.array((x_max - x_min, y_max - y_min, z_max - z_min))

xyz_min = np.array((x_min, y_min, z_min))
xyz_max = np.array((x_max, y_max, z_max))

step_2nm = 2

# %% histograms parameters
bins_total = np.array(np.hstack((xyz_min.reshape(3, 1), xyz_max.reshape(3, 1))))

x_bins_2nm = np.arange(x_min, x_max + 1, step_2nm)
y_bins_2nm = np.arange(y_min, y_max + 1, step_2nm)
z_bins_2nm = np.arange(z_min, z_max + 1, step_2nm)

bins_2nm = [x_bins_2nm, y_bins_2nm, z_bins_2nm]
hist_2nm_shape = (len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1)

nm3_to_cm_3 = 1e-7 ** 3
uint16_max = np.iinfo(np.uint16).max
uint32_max = np.iinfo(np.uint32).max

# %% resist_matrix
n_chain_pos = 0
beg_mon, mid_mon, end_mon = 0, 1, 2
free_mon, free_rad_mon = 10, 20

# %% chain_table
x_pos, y_pos, z_pos = 0, 1, 2
mon_line_pos_pos = 3
mon_type_pos = -1
