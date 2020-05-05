import numpy as np

# %% mapping Harris matrices
x_min, x_max = -50, 50
y_min, y_max = -50, 50
z_min, z_max = 0, 500

l_xyz = np.array((x_max - x_min, y_max - y_min, z_max - z_min))

xyz_min = np.array((x_min, y_min, z_min))
xyz_max = np.array((x_max, y_max, x_max))

step_10nm = 10
step_2nm = 2

# %% histograms parameters
bins_total = np.array(np.hstack((xyz_min.reshape(3, 1), xyz_max.reshape(3, 1))))

x_bins_10nm = np.arange(x_min, x_max + 1, step_10nm)
y_bins_10nm = np.arange(y_min, y_max + 1, step_10nm)
z_bins_10nm = np.arange(z_min, z_max + 1, step_10nm)

x_bins_2nm = np.arange(x_min, x_max + 1, step_2nm)
y_bins_2nm = np.arange(y_min, y_max + 1, step_2nm)
z_bins_2nm = np.arange(z_min, z_max + 1, step_2nm)

bins_10nm = [x_bins_10nm, y_bins_10nm, z_bins_10nm]
bins_2nm = [x_bins_2nm, y_bins_2nm, z_bins_2nm]
