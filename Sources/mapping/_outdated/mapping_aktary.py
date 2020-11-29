import numpy as np

# %% _outdated harris matrices
x_min, x_max = -50, 50
y_min, y_max = -50, 50
z_min, z_max = 0, 100

l_xyz = np.array((x_max - x_min, y_max - y_min, z_max - z_min))
l_x, l_y, l_z = l_xyz

xyz_min = np.array((x_min, y_min, z_min))
xyz_max = np.array((x_max, y_max, z_max))

step_05nm = 0.5
step_1nm = 1
step_2nm = 2
step_4nm = 4
step_5nm = 5

d_PMMA_nm = z_max
d_PMMA_cm = d_PMMA_nm * 1e-7
area_cm2 = (x_max - x_min) * (y_max - y_min) * 1e-7 ** 2
volume_cm3 = area_cm2 * d_PMMA_cm

# %% histograms parameters
x_bins_05nm = np.arange(x_min, x_max + 0.1, step_05nm)
y_bins_05nm = np.arange(y_min, y_max + 0.1, step_05nm)
z_bins_05nm = np.arange(z_min, z_max + 0.1, step_05nm)

x_bins_1nm = np.arange(x_min, x_max + 1, step_1nm)
y_bins_1nm = np.arange(y_min, y_max + 1, step_1nm)
z_bins_1nm = np.arange(z_min, z_max + 1, step_1nm)

x_bins_2nm = np.arange(x_min, x_max + 1, step_2nm)
y_bins_2nm = np.arange(y_min, y_max + 1, step_2nm)
z_bins_2nm = np.arange(z_min, z_max + 1, step_2nm)

x_bins_4nm = np.arange(x_min, x_max + 1, step_4nm)
y_bins_4nm = np.arange(y_min, y_max + 1, step_4nm)
z_bins_4nm = np.arange(z_min, z_max + 1, step_4nm)

x_bins_5nm = np.arange(x_min, x_max + 1, step_5nm)
y_bins_5nm = np.arange(y_min, y_max + 1, step_5nm)
z_bins_5nm = np.arange(z_min, z_max + 1, step_5nm)

bins_05nm = [x_bins_05nm, y_bins_05nm, z_bins_05nm]
bins_1nm = [x_bins_1nm, y_bins_1nm, z_bins_1nm]
bins_2nm = [x_bins_2nm, y_bins_2nm, z_bins_2nm]
bins_4nm = [x_bins_4nm, y_bins_4nm, z_bins_4nm]
bins_5nm = [x_bins_5nm, y_bins_5nm, z_bins_5nm]

hist_05nm_shape = (len(x_bins_05nm) - 1, len(y_bins_05nm) - 1, len(z_bins_05nm) - 1)
hist_1nm_shape = (len(x_bins_1nm) - 1, len(y_bins_1nm) - 1, len(z_bins_1nm) - 1)
hist_2nm_shape = (len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1)
hist_4nm_shape = (len(x_bins_4nm) - 1, len(y_bins_4nm) - 1, len(z_bins_4nm) - 1)
hist_5nm_shape = (len(x_bins_5nm) - 1, len(y_bins_5nm) - 1, len(z_bins_5nm) - 1)
