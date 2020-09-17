# %% cutoff energy indexes
PMMA_E_cut_ind = 1
Si_E_cut_ind = 278  # plasmon energy

# %% simulation indexes
vacuum_ind = 2
PMMA_ind = 0
Si_ind = 1

sim_elastic_ind = 0
sim_PMMA_ee_val_ind = 1
sim_PMMA_phonon_ind = 4
sim_PMMA_polaron_ind = 5

sim_MuElec_plasmon_ind = 0

# %% DATA indexes
DATA_e_id_ind = 0
DATA_parent_e_id_ind = 1
DATA_layer_id_ind = 2
DATA_process_id_ind = 3
DATA_x_ind = 4
DATA_y_ind = 5
DATA_z_ind = 6
DATA_E_dep_ind = 7
DATA_E2nd_ind = 8
DATA_E_ind = 9
DATA_xy_inds = range(DATA_x_ind, DATA_y_ind + 1)
DATA_coord_inds = range(DATA_x_ind, DATA_z_ind + 1)

DATA_line_len = 10

# %% resist_matrix
n_chain_ind = 0
begin_monomer, middle_monomer, end_monomer = 0, 1, 2
free_monomer, gone_monomer = 10, 20

# %% chain_table
x_pos, y_pos, z_pos = 0, 1, 2
monomer_type_ind = -1
