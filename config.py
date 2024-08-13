len_trajectories = 10
nb_trajectories = 172
train_set_pct = 0.5
val_set_pct = 0.3
system_configuration = {
    'dof2':{
            'state_variable':['vy', 'psidt'],
            'control_variable':['vx', 'alpha1']
        },
    'duffing':{
            'state_variable':['x', 'y'],
            'control_variable':['f']
        }
}
device = 'cuda'
smoothing_parameters = {}