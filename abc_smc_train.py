import numpy as np
import pandas as pd
import torch
import os
from utils import to_numpy, SimpleDataLoader, save_to_json, data_preprocessing, data_preprocessing_, open_data, split_data, create_folder_if_not_exists
# from vehicle_dynamic import vehicle_parameters
from config import *
# import metrics as metrics
from tqdm import tqdm
import gc
from nn_architecture import TorchDOF2, duffing_oscillator
from scipy.stats import norm
import argparse

class ABCSmc:

        def __init__(self, targeted_acceptance_rate, nb_samples):
            # self.thresholds = thresholds
            self.nb_samples = nb_samples
            self.nb_populations = len(targeted_acceptance_rate)
            self.targeted_acceptance_rate = targeted_acceptance_rate
        
        def run(self, data, model, std, nb_integration_steps):

            self.parameters_name = list(model.get_parameters().keys())

            prior_probs = np.zeros(self.nb_samples)
            samples = np.zeros((self.nb_samples, len(self.parameters_name)))
            distances = np.zeros(self.nb_samples)
            weights = np.ones(self.nb_samples)
            samples_prev = np.ones(self.nb_samples)
            self.populations = []
            self.criterions = []

            threshold = 1.2
            current_acceptance_rate = 100

            for pop_idx in range(self.nb_populations):

                self.ctr = 0
                rolling_ctr = 0
                accept = False
                warmup = False


                print()
                print(f'Population Index : {pop_idx} / {self.nb_populations}')

                for i in range(int(1e10)):

                    if pop_idx == 0:
                        
                        # Sample from prior
                        model.sample()
                        model = model.to(device)
                        s = np.asarray(list(model.get_parameters(standardize=True).values()))
                        
                        # Compute distance
                        dist = []

                        for X_in, U_in, X_out, U_out in data:

                            X_pred_list = []

                            # Initial prediction
                            X_out_pred = X_in[:, 0] + model(X_in[:, 0], U_in[:, 0])

                            # Loop through the remaining residual blocks
                            for pred_idx in range(nb_integration_steps):
                                X_pred_list.append(X_out_pred.clone())
                                X_out_pred = X_out_pred + model(X_out_pred, U_out[:, pred_idx])
                                
                            # Stack the predictions along the second dimension to form the final X_pred tensor
                            X_pred = torch.stack(X_pred_list, dim=1).to(device)
                            dist.append(torch.sqrt(torch.mean(((X_out - X_pred) / std)**2)).detach().item())
                        
                        dist = np.mean(dist)
                                                
                        # Reject sample if greater than specified thresholds
                        if dist < threshold: #self.thresholds[pop_idx]:
                            if accept:
                                samples[self.ctr] = s
                                distances[self.ctr] = dist
                                self.ctr += 1
                            rolling_ctr += 1

                    elif pop_idx > 0:

                        # Select individual and compute pertubation
                        sampling_idx = np.random.choice(np.arange(self.nb_samples), size=1, replace=True, p=weights_prev)[0]
                        s = np.random.normal(samples_prev[sampling_idx], 0.2)
                        
                        model.set_parameters(self.parameters_name, s)
                        model = model.cuda()

                        # Compute prior probability
                        prior_probs[self.ctr] = self.get_prior_probs(torch.tensor(s))

                        if prior_probs[self.ctr] != 0:
                            
                            # Compute distance
                            dist = []
                            for X_in, U_in, X_out, U_out in data:
       
                                X_pred_list = []

                                # Initial prediction
                                X_out_pred = X_in[:, 0] + model(X_in[:, 0], U_in[:, 0])

                                # Loop through the remaining residual blocks
                                for pred_idx in range(nb_integration_steps):
                                    X_pred_list.append(X_out_pred.clone())
                                    X_out_pred = X_out_pred + model(X_out_pred, U_out[:, pred_idx])
                                    
                                # Stack the predictions along the second dimension to form the final X_pred tensor
                                X_pred = torch.stack(X_pred_list, dim=1).to(device)
                                dist.append(torch.sqrt(torch.mean(((X_out - X_pred) / std)**2)).detach().item())

                            dist = np.mean(dist)
                            # Reject sample if greater than specified thresholds
                            if dist < threshold: #self.thresholds[pop_idx]:
                                if accept:
                                    samples[self.ctr] = s
                                    distances[self.ctr] = dist
                                    self.ctr += 1
                                rolling_ctr += 1

                    if i % 20 == 0:
                        print(f"- Nb samples accepted {self.ctr} | Rejected {i - self.ctr} | current_acceptance : {current_acceptance_rate} | {threshold}", end='\r', flush=True) # | Acceptation rate : {(self.ctr / (i + 1e-8)) * 100}
                    
                    if i % 1000 == 0 and i > 0 and warmup == False:
                        current_acceptance_rate = (rolling_ctr / 1000) * 100
                        if np.abs(current_acceptance_rate - self.targeted_acceptance_rate[pop_idx]) > 1.99:
                            if current_acceptance_rate > self.targeted_acceptance_rate[pop_idx]:
                                threshold = threshold * 0.95
                            elif current_acceptance_rate < self.targeted_acceptance_rate[pop_idx]:
                                threshold = threshold * 1.05
                                accept = False
                        else:
                            accept = True
                            warmup = True

                        rolling_ctr = 0

                    if self.ctr == self.nb_samples:
                        break

                # Compute weight
                if pop_idx > 0:
                    print()
                    for i in range(self.nb_samples):
                        print(f"- Compute weights : {(i / (self.nb_samples + 1e-8)) * 100} %", end='\r', flush=True)
                        weights[i] = prior_probs[i] / np.sum([weights_prev[j] * norm.pdf(samples[i], samples_prev[j], 0.1) for j in range(len(samples_prev))])

                # Copy previous weights
                weights_prev = weights.copy()

                # Normalize weights
                weights_prev /= np.sum(weights_prev)
                samples_prev = samples.copy()
                self.populations.append(samples_prev)
                self.criterions.append(distances)
            
            return self.populations, self.criterions
        
        def get_prior_probs(self, x):
            mean = torch.tensor([0] * len(self.parameters_name))
            std = torch.tensor([1] * len(self.parameters_name))
            dist = torch.distributions.normal.Normal(mean, std)
            pdf_value = torch.exp(torch.sum(dist.log_prob(x)))
            return pdf_value
  
        def save_results(self):
            return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")

    # Add command-line options
    parser.add_argument("--training_name", type=str, help="Training name")
    parser.add_argument("--training_parameters", type=str, help="Training parameters")

    # # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the options
    training_name = args.training_name
    training_parameters = args.training_parameters
    
    start_idx = 0
    end_idx = 1
    train = True
    test = True

    training_folder = f"./results/training_{training_name}/"
    parameters_path = f"./training_parameters/{training_parameters}.csv"

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
    parameters = pd.read_csv(parameters_path)
    parameters.to_csv(f'{training_folder}/config.csv')
    
    assert end_idx <= len(parameters)

    for training_idx in range(start_idx, end_idx):
        
        # Open training parameters
        dataset_choice = str(parameters['data'].values[training_idx])
        subsampling_dt = float(parameters['subsampling_dt'].values[training_idx])
        batchsize = int(parameters['batchsize'].values[training_idx])
        future_sequence_duration = float(parameters['future_temporal_horizon'].values[training_idx])
        future_delay = float(parameters['future_delay'].values[training_idx])
        state_configuration = str(parameters['state_configuration'].values[training_idx])
        smoothing = bool(parameters['smoothing'].values[training_idx])
        training_data = str(parameters['data'].values[training_idx])
        seed = int(parameters['seed'].values[training_idx])
        standardize = True
        shuffle = True
        model = str(parameters['state_configuration'].values[training_idx])
        
        data_dt = float(parameters['data_dt'].values[training_idx])

        np.random.seed(seed)

        if dataset_choice == 'february_2024':
            data_path = f'./data/february_2024_bis/final/'
        else:
            data_path = f'./data/duffing_oscillator/'

        data_list = []
        for filename in tqdm(os.listdir(f'{data_path}')):
            data_list.append(pd.read_csv(f'{data_path}{filename}'))

        (train_data_list, train_trajectories_idx, 
         val_data_list, val_trajectories_idx, 
         test_data_list, test_trajectories_idx) = split_data(data_list, nb_trajectories, shuffle, train_set_pct, val_set_pct)

        nb_integration_steps = int(future_sequence_duration / future_delay)

        in_variables = system_configuration[state_configuration]['state_variable']
        out_variables = system_configuration[state_configuration]['state_variable']
        control_variables = system_configuration[state_configuration]['control_variable']
        state_variables = system_configuration[state_configuration]['state_variable']

        if model == 'dof2':
            model = TorchDOF2(
                parameters_path = './nn_architecture/parameters.json', 
                dt = subsampling_dt,
                output_format = 'speed',
                seed=training_idx)
            model.to(device)
        elif model == 'duffing':
            model = duffing_oscillator(
                parameters_path = './nn_architecture/parameters_duffing.json', 
                dt = subsampling_dt,
                output_format = 'speed',
                seed=training_idx)
            model.to(device)

        train_in_state, train_in_control, train_out_state = data_preprocessing(
            data_list = train_data_list.copy(),
            data_dt = data_dt,
            subsampling_dt = subsampling_dt,
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = False,
            smoothing_parameters = smoothing_parameters
        )

        # Instantiate dataloader
        trainDataLoader = SimpleDataLoader(
            train_in_states = train_in_state.copy(),
            train_in_controls = train_in_control.copy(),
            train_out_states = train_out_state.copy(),
            batchsize = batchsize,
            past_sequence_duration = 0.,
            future_sequence_duration = future_sequence_duration,
            past_delay = 0.,
            future_delay = future_delay,
            dt = subsampling_dt,
            shuffle = shuffle,
            device = device
        )

        reshaped_train_in_state = np.asarray(train_in_state).reshape(-1, np.asarray(train_in_state).shape[2])
        reshaped_train_out_state = np.asarray(train_out_state).reshape(-1, np.asarray(train_out_state).shape[2])
        reshaped_train_in_control = np.asarray(train_in_control).reshape(-1, np.asarray(train_in_control).shape[2])
      
        std = reshaped_train_in_state.std(axis=0)
        u_std = reshaped_train_in_control.std(axis=0)

        std = torch.tensor([std]).float().requires_grad_(False).to(device).unsqueeze(1)

        if train:

            abc = ABCSmc(targeted_acceptance_rate = [75, 65, 55, 45, 35, 25, 15, 5, 2.5, 1.], nb_samples = 1000)
            res, crit = abc.run(trainDataLoader, model, std, nb_integration_steps)
            idx = np.argsort(crit[-1])
            for training_idx, r in enumerate(res[-1][idx]):

                model.set_parameters(list(model.get_parameters().keys()), r)
                model.to(device)
                torch.save(model.state_dict(), f'{training_folder}/best_model_{training_idx}.pt')
            
        if test:

            test_in_state, test_in_control, test_out_state, test_out_control = data_preprocessing_(
                data_list = test_data_list.copy(),
                data_dt = data_dt,
                subsampling_dt = subsampling_dt,
                state_variables = in_variables,
                out_variables = out_variables,
                control_variables = control_variables,
                differentiate = False,
                smoothing = False,
                smoothing_parameters = smoothing_parameters
            )

            parameters = {key:[]for key in list(model.get_parameters().keys())}
            for training_idx in tqdm(range(1000)):
                model.load_state_dict(torch.load(f'{training_folder}/best_model_{training_idx}.pt'))
                model.to(device)
                model.eval()
                for key in parameters.keys():
                    parameters[key].append(model.get_parameters()[key])

            create_folder_if_not_exists(f'{training_folder}/results/')
            save_to_json(
                parameters,
                f'{training_folder}/results/parameter.json'
            )

            for training_idx in tqdm(range(1000)):
                
                model.load_state_dict(torch.load(f'{training_folder}/best_model_{training_idx}.pt'))
                model.to(device)
                model.eval()

                nb_test_trajectories = len(test_in_state)
                sequence_duration = 10
                s = 0
                nb_step = int(len_trajectories / subsampling_dt) - 1 - s
                
                X_in_tensor = torch.zeros((nb_test_trajectories, nb_step, len(in_variables))).to(device)
                U_in_tensor = torch.zeros((nb_test_trajectories, nb_step, len(control_variables))).to(device)
                X_out_tensor = torch.zeros((nb_test_trajectories, nb_step, len(out_variables))).to(device)
                X_predictions_tensor = torch.zeros((nb_test_trajectories, nb_step, len(out_variables))).to(device)
                    
                # Fill tensor test_in_state, test_in_control, test_out_state
                for idx, (X_in, X_out, U_in) in enumerate(zip(test_in_state, test_out_state, test_in_control)):
                    X_in_tensor[idx] = torch.tensor(X_in[s:])
                    X_out_tensor[idx] = torch.tensor(X_out[s:])
                    U_in_tensor[idx] = torch.tensor(U_in[s:])
                    
                pred = X_in_tensor[: ,0] + model(X_in_tensor[: ,0], U_in_tensor[:, 0])
                X_predictions_tensor[:, 0, :] = pred.clone()
                for step in range(1, nb_step):
                    pred = pred + model(pred, U_in_tensor[:, step])
                    X_predictions_tensor[:, step, :] = pred.clone()

                times =  [0.05, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                rmse = {}
                for element in ['vy', 'dpsidt', 'rmse']:
                    for time in times:
                        rmse[f'{element}_{time}'] = []

                for time in times:
                    idx_limit = int(time/10 * X_predictions_tensor.shape[1]+1)
                    RMSE = torch.mean(torch.sqrt(torch.mean(((X_predictions_tensor[:, :idx_limit] - X_out_tensor[:, :idx_limit]))**2, axis=1)), axis=0)
                    rmse[f'vy_{time}'] = RMSE[0].item()
                    rmse[f'dpsidt_{time}'] = RMSE[1].item()
                    rmse[f'rmse_{time}'] = RMSE.mean().item()

                save_to_json(
                        rmse,
                        f'{training_folder}/results/rmse_{training_idx}.json'
                    )

            del model  
