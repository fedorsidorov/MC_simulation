import numpy as np

# %%
x_min, x_max = -50, 50
y_min, y_max = -50, 50
z_min, z_max = 0, 900

l_xyz = np.array((x_max - x_min, y_max - y_min, z_max - z_min))
lx, ly, lz = l_xyz

xyz_min = np.array((x_min, y_min, z_min))
xyz_max = np.array((x_max, y_max, z_max))

step_5nm = 5
step_10nm = 10
step_25nm = 25
step_50nm = 50
step_100nm = 100

d_PMMA = z_max
d_PMMA_cm = d_PMMA * 1e-7
area_cm2 = (x_max - x_min) * (y_max - y_min) * 1e-7 ** 2
volume_cm3 = area_cm2 * d_PMMA_cm

# %% histograms parameters
x_bins_5nm = np.arange(x_min, x_max + 1, step_5nm)
y_bins_5nm = np.arange(y_min, y_max + 1, step_5nm)
z_bins_5nm = np.arange(z_min, z_max + 1, step_5nm)

x_bins_10nm = np.arange(x_min, x_max + 1, step_10nm)
y_bins_10nm = np.arange(y_min, y_max + 1, step_10nm)
z_bins_10nm = np.arange(z_min, z_max + 1, step_10nm)

x_bins_25nm = np.arange(x_min, x_max + 1, step_25nm)
y_bins_25nm = np.arange(y_min, y_max + 1, step_25nm)
z_bins_25nm = np.arange(z_min, z_max + 1, step_25nm)

# x_bins_50nm = np.arange(x_min, x_max + 1, step_50nm)
# y_bins_50nm = np.arange(y_min, y_max + 1, step_50nm)
# z_bins_50nm = np.arange(z_min, z_max + 1, step_50nm)

# x_bins_100nm = np.arange(x_min, x_max + 1, step_100nm)
# y_bins_100nm = np.arange(y_min, y_max + 1, step_100nm)
# z_bins_100nm = np.arange(z_min, z_max + 1, step_100nm)

x_centers_5nm = (x_bins_5nm[:-1] + x_bins_5nm[1:]) / 2
y_centers_5nm = (y_bins_5nm[:-1] + y_bins_5nm[1:]) / 2
z_centers_5nm = (z_bins_5nm[:-1] + z_bins_5nm[1:]) / 2

x_centers_10nm = (x_bins_10nm[:-1] + x_bins_10nm[1:]) / 2
y_centers_10nm = (y_bins_10nm[:-1] + y_bins_10nm[1:]) / 2
z_centers_10nm = (z_bins_10nm[:-1] + z_bins_10nm[1:]) / 2

x_centers_25nm = (x_bins_25nm[:-1] + x_bins_25nm[1:]) / 2
y_centers_25nm = (y_bins_25nm[:-1] + y_bins_25nm[1:]) / 2
z_centers_25nm = (z_bins_25nm[:-1] + z_bins_25nm[1:]) / 2

# x_centers_50nm = (x_bins_50nm[:-1] + x_bins_50nm[1:]) / 2
# y_centers_50nm = (y_bins_50nm[:-1] + y_bins_50nm[1:]) / 2
# z_centers_50nm = (z_bins_50nm[:-1] + z_bins_50nm[1:]) / 2

# x_centers_100nm = (x_bins_100nm[:-1] + x_bins_100nm[1:]) / 2
# y_centers_100nm = (y_bins_100nm[:-1] + y_bins_100nm[1:]) / 2
# z_centers_100nm = (z_bins_100nm[:-1] + z_bins_100nm[1:]) / 2

bins_5nm = [x_bins_5nm, y_bins_5nm, z_bins_5nm]
bins_10nm = [x_bins_10nm, y_bins_10nm, z_bins_10nm]
bins_25nm = [x_bins_25nm, y_bins_25nm, z_bins_25nm]
# bins_50nm = [x_bins_50nm, y_bins_50nm, z_bins_50nm]
# bins_100nm = [x_bins_100nm, y_bins_100nm, z_bins_100nm]

hist_5nm_shape = (len(x_bins_5nm) - 1, len(y_bins_5nm) - 1, len(z_bins_5nm) - 1)
hist_10nm_shape = (len(x_bins_10nm) - 1, len(y_bins_10nm) - 1, len(z_bins_10nm) - 1)
hist_25nm_shape = (len(x_bins_25nm) - 1, len(y_bins_25nm) - 1, len(z_bins_25nm) - 1)
# hist_50nm_shape = (len(x_bins_50nm) - 1, len(y_bins_50nm) - 1, len(z_bins_50nm) - 1)
# hist_100nm_shape = (len(x_bins_100nm) - 1, len(y_bins_100nm) - 1, len(z_bins_100nm) - 1)
