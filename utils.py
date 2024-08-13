import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import torch
from scipy.signal import savgol_filter
import os

def subsample_dataframe(df, initial_dt, new_dt):
        subsampling_factor = int(new_dt / initial_dt)
        return df.iloc[np.arange(0, len(df), subsampling_factor).astype('int')]

def get_lagged_sequence(X, nb_lag):
    sequence_size = X.shape[0]
    lagged_X = np.zeros((sequence_size - nb_lag, nb_lag, X.shape[1]))
    for var_idx in range(X.shape[1]):
        for lag in range(nb_lag):
            lagged_X[:sequence_size - nb_lag, lag, var_idx] = X[lag:sequence_size - nb_lag+lag, var_idx]
    return lagged_X

def data_preprocessing(
        data_list, 
        data_dt, 
        subsampling_dt, 
        state_variables, 
        control_variables,
        out_variables,
        differentiate=False,
        difference=False,
        smoothing=False,
        smoothing_parameters = dict(),
        use_smoothing_derivatives = False
    ):

    train_in_state = []
    train_in_control = []
    train_out_state = []

    for data in data_list:

        if smoothing:

            for variable in state_variables:
                data[variable] = savgol_filter(
                                    data[variable].values, 
                                    window_length = smoothing_parameters[variable][0],
                                    polyorder = smoothing_parameters[variable][1]
                                )
            
        data = subsample_dataframe(data, data_dt, subsampling_dt)

        in_state = data[state_variables].values[:-1]
        in_control = data[control_variables].values[:-1]

        if not differentiate:
            out_state = data[out_variables].values[:-1]
        elif differentiate:
            out_state = (data[out_variables].values[1:] - data[out_variables].values[:-1]) / subsampling_dt
        elif difference:
            out_state = data[out_variables].values[1:] - data[out_variables].values[:-1]

        train_in_state.append(in_state)
        train_in_control.append(in_control)
        train_out_state.append(out_state)

    return train_in_state, train_in_control, train_out_state

def data_preprocessing_(
        data_list, 
        data_dt, 
        subsampling_dt, 
        state_variables, 
        control_variables,
        out_variables,
        differentiate=False,
        difference=False,
        smoothing=False,
        smoothing_parameters = dict(),
        use_smoothing_derivatives = False
    ):

    train_in_state = []
    train_in_control = []
    train_out_state = []
    train_out_control = []
    
    for data in data_list:

        if smoothing:

            for variable in state_variables:
                data[variable] = savgol_filter(
                                    data[variable].values, 
                                    window_length = smoothing_parameters[variable][0],
                                    polyorder = smoothing_parameters[variable][1]
                                )
            
        data = subsample_dataframe(data, data_dt, subsampling_dt)

        in_state = data[state_variables].values[:-1]
        in_control = data[control_variables].values[:-1]

        if not differentiate:
            out_state = data[out_variables].values[1:]
            out_control = data[control_variables].values[1:]
        elif differentiate:
            out_state = (data[out_variables].values[1:] - data[out_variables].values[:-1]) / subsampling_dt
        elif difference:
            out_state = data[out_variables].values[1:] - data[out_variables].values[:-1]

        train_in_state.append(in_state)
        train_in_control.append(in_control)
        train_out_state.append(out_state)
        train_out_control.append(out_control)

    return train_in_state, train_in_control, train_out_state, train_out_control

def open_data(data_path, file_name, nb_trajectories):
    data_list = []
    for idx in tqdm(np.arange(0, nb_trajectories)):
        data_list.append(pd.read_csv(f'{data_path}/{file_name}_{idx}.csv'))
    return data_list

def split_data(data_list, nb_trajectories, shuffle, train_set_pct, val_set_pct):
    trajectories_idx = np.arange(0, nb_trajectories)
    if shuffle :
        np.random.shuffle(trajectories_idx) #[:int((train_set_pct + val_set_pct + test_trajectories_idx) * nb_trajectories)])
    train_trajectories_idx = trajectories_idx[:int(train_set_pct * nb_trajectories)]
    val_trajectories_idx = trajectories_idx[int(train_set_pct * nb_trajectories):int((train_set_pct + val_set_pct) * nb_trajectories)]
    test_trajectories_idx = trajectories_idx[int((train_set_pct + val_set_pct) * nb_trajectories):]
    train_data_list, val_data_list, test_data_list = [], [], []
    for idx in train_trajectories_idx:
        train_data_list.append(data_list[idx].copy())
    for idx in val_trajectories_idx:
        val_data_list.append(data_list[idx].copy())
    for idx in test_trajectories_idx:
        test_data_list.append(data_list[idx].copy())
    return train_data_list, train_trajectories_idx, val_data_list, val_trajectories_idx, test_data_list, test_trajectories_idx

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def save_to_json(data, file_name):
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)

def open_json(path):
    f = open(path, 'r')
    data = json.loads(f.read())
    f.close()
    return data

class SimpleDataLoader:

    def __init__(
            self, 
            train_in_states,
            train_out_states,
            train_in_controls,
            batchsize,
            past_sequence_duration,
            future_sequence_duration,
            past_delay,
            future_delay,
            dt,
            shuffle,
            device,
            ):
    
        self.batchsize = batchsize
        self.nb_in_state = train_in_states[0].shape[-1]
        self.nb_out_state = train_out_states[0].shape[-1]
        self.nb_control = train_in_controls[0].shape[-1]

        self.past_sequence_duration = past_sequence_duration
        self.past_delay = past_delay

        self.future_sequence_duration = future_sequence_duration
        self.future_delay = future_delay

        if past_sequence_duration == 0:
            nb_element_per_past_sequence = 1
            past_sequence_size = 1
            space_between_past_element = 0
            past_element_idx = np.asarray([0])
        else:
            nb_element_per_past_sequence = int(past_sequence_duration / past_delay) + 1 # add 1 for sample corresponding to t = 0 sample
            past_sequence_size = int(past_sequence_duration / dt) + 1
            space_between_past_element = int(past_delay / dt)
            past_element_idx = np.arange(0, past_sequence_size, space_between_past_element).astype('int')

        if future_sequence_duration == 0:
            nb_element_per_future_sequence = 1 # int(future_sequence_duration / future_delay) # add 1 for sample corresponding to t = 0 sample
            future_sequence_size = 1 # int(future_sequence_duration / dt)
            space_between_future_element = 0 # int(future_delay / dt)
            future_element_idx = np.asarray([1]) # np.arange(1, future_sequence_size + 1, space_between_future_element).astype('int')
        else:
            nb_element_per_future_sequence = int(future_sequence_duration / future_delay) # add 1 for sample corresponding to t = 0 sample
            future_sequence_size = int(future_sequence_duration / dt)
            space_between_future_element = int(future_delay / dt)
            future_element_idx = np.arange(1, future_sequence_size + 1, space_between_future_element).astype('int')
       
        assert batchsize < len(train_in_states[0]) * train_in_states[1].shape[0]

        X_in_state, X_in_control, X_out_state, X_out_control = [], [], [], []
        for train_in_state, train_in_control, train_out_state in zip(train_in_states, train_in_controls, train_out_states):

            for idx in range(0, train_in_state.shape[0] - past_sequence_size - future_sequence_size, 1):
                in_idx = idx + past_element_idx
                out_idx = in_idx[-1] + future_element_idx
                X_in_state.append(train_in_state[in_idx])
                X_in_control.append(train_in_control[in_idx])
                X_out_state.append(train_out_state[out_idx])
                X_out_control.append(train_in_control[out_idx])

        X_in_state, X_in_control, X_out_state, X_out_control = np.asarray(X_in_state), np.asarray(X_in_control), np.asarray(X_out_state), np.asarray(X_out_control)

        if shuffle:
            X_in_state, X_in_control, X_out_state, X_out_control = self.shuffle_sequence(X_in_state, X_in_control, X_out_state, X_out_control)

        nb_batch = len(X_in_state) // batchsize
        if len(X_in_state) % batchsize != 0:
            X_in_state = X_in_state[:-1].copy()
            X_in_control = X_in_control[:-1].copy()
            X_out_state = X_out_state[:-1].copy()
            X_out_control = X_out_control[:-1].copy()

        self.batch_X_in_state = np.zeros((nb_batch, batchsize, nb_element_per_past_sequence, self.nb_in_state))
        self.batch_X_in_control = np.zeros((nb_batch, batchsize, nb_element_per_past_sequence, self.nb_control))
        self.batch_X_out_state = np.zeros((nb_batch, batchsize, nb_element_per_future_sequence, self.nb_out_state))
        self.batch_X_out_control = np.zeros((nb_batch, batchsize, nb_element_per_future_sequence, self.nb_control))
        
        idx = 0
        for batch_idx in range(nb_batch):
            for sequence_idx in range(batchsize):
                self.batch_X_in_state[batch_idx, sequence_idx] = X_in_state[idx].copy()
                self.batch_X_in_control[batch_idx, sequence_idx] = X_in_control[idx].copy()
                self.batch_X_out_state[batch_idx, sequence_idx] = X_out_state[idx].copy()
                self.batch_X_out_control[batch_idx, sequence_idx] = X_out_control[idx].copy()
                idx += 1

        del X_in_state, X_in_control, X_out_state, X_out_control

        self.batch_X_in_state = torch.tensor(self.batch_X_in_state).float().to(device)
        self.batch_X_in_control = torch.tensor(self.batch_X_in_control).float().to(device)
        self.batch_X_out_state = torch.tensor(self.batch_X_out_state).float().to(device)
        self.batch_X_out_control = torch.tensor(self.batch_X_out_control).float().to(device)
        
    def shuffle_sequence(self, X, Y, Z, U):
        shuffled_idx = np.arange(X.shape[0])
        np.random.shuffle(shuffled_idx)
        return X[shuffled_idx], Y[shuffled_idx], Z[shuffled_idx], U[shuffled_idx]
    
    def __len__(self):
        return self.batch_X_in_state.shape[0]

    def __getitem__(self, index):       
        return self.batch_X_in_state[index], self.batch_X_in_control[index], self.batch_X_out_state[index], self.batch_X_out_control[index]
     
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")