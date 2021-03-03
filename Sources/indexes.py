# %% cutoff energy indexes
# PMMA_E_cut_ind = 1
PMMA_E_cut_ind = 135  # PMMA work function

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

# %% e_DATA indexes
e_DATA_e_id_ind = 0
e_DATA_parent_e_id_ind = 1
e_DATA_layer_id_ind = 2
e_DATA_process_id_ind = 3
e_DATA_x_ind = 4
e_DATA_y_ind = 5
e_DATA_z_ind = 6
e_DATA_E_dep_ind = 7
e_DATA_E2nd_ind = 8
e_DATA_E_ind = 9
e_DATA_xy_inds = range(e_DATA_x_ind, e_DATA_y_ind + 1)
e_DATA_coord_inds = range(e_DATA_x_ind, e_DATA_z_ind + 1)

e_DATA_line_len = 10

# %% resist_matrix
n_chain_ind = 0
begin_monomer, middle_monomer, end_monomer = 0, 1, 2
free_monomer, gone_monomer = 10, 20

# %% chain_table
x_pos, y_pos, z_pos = 0, 1, 2
monomer_type_ind = -1
