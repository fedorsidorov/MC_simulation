import numpy as np

# %% _outdated EXP matrices
x_min, x_max = -1500, 1500
y_min, y_max = -10, 10
z_min, z_max = 0, 80

l_xyz = np.array((x_max - x_min, y_max - y_min, z_max - z_min))
l_x, l_y, l_z = l_xyz

xyz_min = np.array((x_min, y_min, z_min))
xyz_max = np.array((x_max, y_max, z_max))

step_2nm = 2
step_5nm = 5

d_PMMA_nm = z_max
d_PMMA_cm = d_PMMA_nm * 1e-7
area_cm2 = (x_max - x_min) * (y_max - y_min) * 1e-7 ** 2
volume_cm3 = area_cm2 * d_PMMA_cm

# %% histograms parameters
x_bins_2nm = np.arange(x_min, x_max + 1, step_2nm)
y_bins_2nm = np.arange(y_min, y_max + 1, step_2nm)
z_bins_2nm = np.arange(z_min, z_max + 1, step_2nm)

x_centers_2nm = (x_bins_2nm[:-1] + x_bins_2nm[1:]) / 2
y_centers_2nm = (y_bins_2nm[:-1] + y_bins_2nm[1:]) / 2
z_centers_2nm = (z_bins_2nm[:-1] + z_bins_2nm[1:]) / 2

x_bins_5nm = np.arange(x_min, x_max + 1, step_5nm)
y_bins_5nm = np.arange(y_min, y_max + 1, step_5nm)
z_bins_5nm = np.arange(z_min, z_max + 1, step_5nm)

x_centers_5nm = (x_bins_5nm[:-1] + x_bins_5nm[1:]) / 2
y_centers_5nm = (y_bins_5nm[:-1] + y_bins_5nm[1:]) / 2
z_centers_5nm = (z_bins_5nm[:-1] + z_bins_5nm[1:]) / 2

bins_2nm = [x_bins_2nm, y_bins_2nm, z_bins_2nm]
bins_5nm = [x_bins_5nm, y_bins_5nm, z_bins_5nm]

hist_2nm_shape = (len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1)
hist_5nm_shape = (len(x_bins_5nm) - 1, len(y_bins_5nm) - 1, len(z_bins_5nm) - 1)
