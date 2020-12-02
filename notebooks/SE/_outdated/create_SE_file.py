#%% Import
import numpy as np
import matplotlib.pyplot as plt


#%% read datafile
# profile = np.loadtxt('notebooks/SE/profile.txt')
profile = np.load('notebooks/Leveder/2010/0.txt')

arr_y_pre = profile[::3, 0]
arr_z_pre = profile[::3, 1] * 5

step_lx = 8
step_ly = 8

dummy_y = np.linspace(-step_ly/2, step_ly/2, 20)
dummy_z = np.ones(len(dummy_y)) * arr_z_pre.max()

inds_1 = np.where(dummy_y < arr_y_pre[0])[0]
inds_2 = np.where(dummy_y > arr_y_pre[-1])[0]

arr_y = np.concatenate((dummy_y[inds_1], arr_y_pre, dummy_y[inds_2]))
arr_z = np.concatenate((dummy_z[inds_1], arr_z_pre, dummy_z[inds_2]))

# plt.figure(dpi=300)
# plt.plot(arr_y, arr_z)
# plt.show()

arr_y_final = np.concatenate((arr_y, arr_y + step_ly, arr_y + step_ly*2, arr_y + step_ly*3))
arr_z_final = np.concatenate((arr_z, arr_z, arr_z, arr_z))

arr_y = arr_y_final
arr_z = arr_z_final

n = len(arr_y)
half_l = step_lx / 2

volume = np.trapz(arr_z, x=arr_y) * step_lx

plt.figure(dpi=300)
plt.plot(arr_y, arr_z)
plt.show()

# %% vertices
V = np.zeros((4*n, 1+3))

for i in range(n):
    V[0*n + i, :] = 100+i,  half_l, arr_y[i], 0
    V[1*n + i, :] = 200+i, -half_l, arr_y[i], 0
    V[2*n + i, :] = 300+i,  half_l, arr_y[i], arr_z[i]
    V[3*n + i, :] = 400+i, -half_l, arr_y[i], arr_z[i]

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.plot(V[:, 1], V[:, 2], V[:, 3], 'o')
plt.show()

# %% edges
E = np.zeros((4*n + 4*(n-1), 1+2))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

for i in range(n):
    
    V_ind_100 = 100+i
    V_ind_200 = 200+i
    V_ind_300 = 300+i
    V_ind_400 = 400+i
    
    E[0*n + i, :] = 1200+i, V_ind_100, V_ind_200
    E[1*n + i, :] = 1300+i, V_ind_100, V_ind_300
    E[2*n + i, :] = 2400+i, V_ind_200, V_ind_400
    E[3*n + i, :] = 3400+i, V_ind_300, V_ind_400

    for j in range(4):
        
        beg_x, beg_y, beg_z = V[np.where(V[:, 0] == E[n*j + i, 1])[0], 1:].reshape((3, 1))
        end_x, end_y, end_z = V[np.where(V[:, 0] == E[n*j + i, 2])[0], 1:].reshape((3, 1))
    
        dx = np.array([beg_x, end_x])[:, 0]
        dy = np.array([beg_y, end_y])[:, 0]
        dz = np.array([beg_z, end_z])[:, 0]
        
        ax.plot(dx, dy, dz)

for i in range(n-1):
    
    for j in range(1, 5):

        E[4*n + (j-1)*(n-1) + i] = 100*int(str(j)*2) + i, 100*j + i, 100*j + i+1
        
        beg_x, beg_y, beg_z = V[np.where(V[:, 0] == 100*j + i  )[0], 1:].reshape((3, 1))
        end_x, end_y, end_z = V[np.where(V[:, 0] == 100*j + i+1)[0], 1:].reshape((3, 1))
        
        dx = np.array([beg_x, end_x])[:, 0]
        dy = np.array([beg_y, end_y])[:, 0]
        dz = np.array([beg_z, end_z])[:, 0]
        
        ax.plot(dx, dy, dz)

plt.show()

# %% faces
F = np.zeros((4*(n-1) + 2, 1+4))

for i in range(n-1):
    
    F[0*(n-1) + i] = 100+i,  (1300 + i+1), -(3300 + i), -(1300 + i),  (1100 + i)
    F[1*(n-1) + i] = 200+i,  (3400 + i+1), -(4400 + i), -(3400 + i),  (3300 + i)
    F[2*(n-1) + i] = 300+i, -(2400 + i+1), -(2200 + i),  (2400 + i),  (4400 + i)
    F[3*(n-1) + i] = 400+i, -(1200 + i+1), -(1100 + i),  (1200 + i),  (2200 + i)

F[4*(n-1), :] = 500, 1300, 3400, -2400, -1200
F[4*(n-1) + 1, :] = 501, -(1300 + n-1), (1200 + n-1), (2400 + n-1), -(3400 + n-1)

# %% datafile
file = ''

file += 'define vertex attribute vmob real\n\n'

file += 'MOBILITY_TENSOR \n' +\
    'vmob  0     0\n' +\
    '0     vmob  0\n' +\
    '0     0  vmob\n\n'

file += 'PARAMETER step_lx = ' + str(step_lx) + '\n\n'
file += 'PARAMETER angle_s_ind = 55' + '\n'
file += 'PARAMETER angle_w_ind = 90' + '\n'

file += 'PARAMETER TENS_r = 33.5e-2' + '\n'
file += 'PARAMETER TENS_s = -TENS_r*cos((angle_s_ind)*pi/180)' + '\n'
file += 'PARAMETER TENS_w = -TENS_r*cos((angle_w_ind)*pi/180)' + '\n\n'

file += '/*--------------------CONSTRAINTS START--------------------*/\n'
file += 'constraint 1 /* fixing the resist on the substrate surface */\n'
file += 'formula: x3 = 0\n\n'
file += 'constraint 13 /* mirror plane, resist on front-side wall */\n'
file += 'formula: x1 = 0.5*step_lx\n\n'
file += 'constraint 24 /* mirror plane, resist on back-side wall */\n'
file += 'formula: x1 = -0.5*step_lx\n'
file += '/*--------------------CONSTRAINTS END--------------------*/\n\n'

# vertices
file += '/*--------------------VERTICES START--------------------*/\n'
file += 'vertices\n'


for i in range(len(V)):
        
    V_ind = str(int(V[i, 0]))
    V_coords = str(V[i, 1:])[1:-1]
    
    file += V_ind + '\t' + V_coords
    
    V0 = V_ind[0]
    
    if V_ind == '100':
        file += '\tconstraints 1 13'
    elif V_ind == '200':
        file += '\tconstraints 1 24'
    elif V_ind == '300':
        file += '\tconstraint 13'
    elif V_ind == '400':
        file += '\tconstraint 24'
    elif V_ind == str(100+n-1):
        file += '\tconstraints 1 13'
    elif V_ind == str(200+n-1):
        file += '\tconstraints 1 24'
    elif V_ind == str(300+n-1):
        file += '\tconstraint 13'
    elif V_ind == str(400+n-1):
        file += '\tconstraint 24'
    elif V0 == '1':
        file += '\tconstraints 1 13'
    elif V0 == '2':
        file += '\tconstraints 1 24'
    elif V0 == '3':
        file += '\tconstraint 13'
    elif V0 == '4':
        file += '\tconstraint 24'

    file += ' vmob 0\n'

file += '/*--------------------VERTICES END--------------------*/\n\n'

# edges
file += '/*--------------------EDGES START--------------------*/\n'
file += 'edges' + '\n'


for i in range(len(E)):
    
    E_ind = str(int(E[i, 0]))
    
    beg = str(int(E[i, 1]))
    end = str(int(E[i, 2]))
    
    file += E_ind + '\t' + beg + ' ' + end
    
    E00 = E_ind[:2]
    
    if E_ind == '1200':
        file += '\tconstraint 1'
    elif E_ind == '1300':
        file += '\tconstraint 13'
    elif E_ind == '2400':
        file += '\tconstraint 24'
    elif E_ind == str(1200+n-1):
        file += '\tconstraint 1'
    elif E_ind == str(1300+n-1):
        file += '\tconstraint 13'
    elif E_ind == str(2400+n-1):
        file += '\tconstraint 24'
    elif E00 == '11':
        file += '\tconstraints 1 13'
    elif E00 == '22':
        file += '\tconstraints 1 24'
    elif E00 in ['13', '33']:
        file += '\tconstraint 13'
    elif E00 in ['24', '44']:
        file += '\tconstraint 24'
    elif E00 == '12':
        file += '\tconstraint 1'

    file += '\n'

file += '/*--------------------EDGES END--------------------*/\n\n'

# faces
file += '/*--------------------FACES START--------------------*/\n'
file += 'faces' + '\n'

for i in range(len(F)):
    
    F_ind = str(int(F[i, 0]))
    F_edges = str(F[i, 1:].astype(int))[1:-1]
    
    file += F_ind + '\t' + F_edges

    if F_ind in ['500', '501']:
        file += '\tcolor green tension TENS_r'

    elif F_ind[0] == '2':
        file += '\tcolor brown tension TENS_r'
    elif F_ind[0] == '4':
        file += '\tcolor magenta tension TENS_s'
    else:
        file += '\tcolor yellow tension TENS_w'

    file += '\n'


file += '/*--------------------FACES END--------------------*/\n\n'

# bodies
file += '/*--------------------BODIES--------------------*/\n'
file += 'bodies' + '\n' + '1' + '\t' + '500 '


for fn in F[:, 0]:

    if str(fn)[0] == '2':
        file += str(int(fn)) + ' '

file += '501\tvolume ' + str(volume)


file += '\n\n/*--------------------SIMULATION--------------------*/\n'
file += 'read\n\n'
file += '/*meshing*/\n'
file += 'meshit := {r 1; area_normalization}\n\n'
file += '/*iteration*/\n'
file += 'loopit := {g50; w 0.0005; w 0.0005; V}\n\n'

# write to file
with open('notebooks/SE/SE_input.txt', 'w') as myfile:
    myfile.write(file)
