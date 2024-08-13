import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from utils import to_numpy, SimpleDataLoader, save_to_json, data_preprocessing, split_data, data_preprocessing_, create_folder_if_not_exists
from config import *
from tqdm import tqdm
import gc
from nn_architecture import Mlp_narx, TorchDOF2, discrepancy #, duffing_oscillator
import argparse

def train_loop(
        model,
        optimizer,
        dataLoader,
        standardize,
        std,
        x_col_std,
        nb_integration_steps,
        X_col_out    
        ):

    mse_loss = []
    col_mse_loss = []

    model.train()

    for X_in, U_in, X_out, U_out in dataLoader:

        if not parameter['train_on_collocation']:   
            # Initialize X_pred as a list
            X_pred_list = []

            # Initial prediction
            X_out_pred = X_in[:, 0] + model(X_in[:, 0], U_in[:, 0])
            # X_pred_list.append(X_out_pred.clone())

            # Loop through the remaining residual blocks
            for pred_idx in range(nb_integration_steps):
                X_pred_list.append(X_out_pred.clone())
                X_out_pred = X_out_pred + model(X_out_pred, U_out[:, pred_idx])
                
            # Stack the predictions along the second dimension to form the final X_pred tensor
            X_pred = torch.stack(X_pred_list, dim=1).to(device)

            # Ensure X_pred requires gradients
            X_pred.requires_grad_()

            if standardize :
                X_out = X_out / std
                X_pred = X_pred / std

            se = torch.mean((X_out - X_pred)**2)
        else:
            se = torch.tensor(0.).to(device)
        if parameter['train_on_collocation']:

            # Initialize X_col_pred as a list
            X_col_pred_list = []

            # Initial prediction
            pred = X_col_in + model(X_col_in, U_col_in)
            X_col_pred_list.append(pred.clone())

            # Stack the predictions along the second dimension to form the final X_col_pred tensor
            X_col_pred = torch.stack(X_col_pred_list, dim=1).to(device)

            # Ensure X_col_pred requires gradients
            X_col_pred.requires_grad_()

            if standardize :
                X_col_out_std = X_col_out / x_col_std
                X_col_pred_std = X_col_pred / x_col_std

            col_se = torch.mean((X_col_out_std - X_col_pred_std)**2)

        else:
            col_se = torch.tensor(0.).to(device)
        if parameter['train_on_collocation']:
            loss = col_se
        else:
            loss = se
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse_loss.append(se.detach().cpu().item())
        col_mse_loss.append(col_se.detach().cpu().item())
    
    mse_loss = np.asarray(mse_loss)
    rmse_loss = np.sqrt(np.mean(mse_loss, axis = 0))

    col_mse_loss = np.asarray(col_mse_loss)
    col_rmse_loss = np.sqrt(np.mean(col_mse_loss, axis = 0))

    return rmse_loss, col_rmse_loss

def validation_loop(
        model,
        valDataLoaderOneStep,
        standardize,
        std,
        nb_integration_steps,
    ):
    
    model.eval()
    loss = []

    for X_in, U_in, X_out, U_out in valDataLoaderOneStep:

        X_pred = torch.zeros(X_out.shape, requires_grad=False).to(device)
        X_out_pred = X_in[:, 0] + model(X_in[:, 0], U_in[:, 0])
        for pred_idx in range(nb_integration_steps): 
            X_pred[:, pred_idx] = X_out_pred.clone().detach()
            X_out_pred = X_out_pred + model(X_out_pred, U_out[:, pred_idx])

        if standardize :
            X_out = X_out / std
            X_pred = X_pred / std

        se = (X_out - X_pred)**2
        loss.append(to_numpy(se))

    loss = np.asarray(loss)
    loss = loss.reshape(loss.shape[0] * loss.shape[1] * loss.shape[2], loss.shape[3])
    loss = np.sqrt(np.mean(loss, axis = 0))

    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")

    # Add command-line options
    parser.add_argument("--training_name", type=str, help="Training name")
    parser.add_argument("--training_parameters", type=str, help="Training parameters")
    parser.add_argument("--start_index", type=int, help="Start training index")
    parser.add_argument("--end_index", type=int, help="End training idx")

    # # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the options
    training_name = args.training_name
    training_parameters = args.training_parameters
    start_idx = args.start_index
    end_idx = args.end_index

    print("Launch traning")

    # training_name = "final_results_bis"
    # training_parameters = "physics_informed"
    # start_idx = 0
    # end_idx = 405

    training_folder = f"./results/training_{training_name}/"
    parameters_path = f"./training_parameters/{training_parameters}.csv"

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
    parameters = pd.read_csv(parameters_path)
    parameters.to_csv(f'{training_folder}/config.csv')
    
    assert end_idx <= len(parameters)

    for training_idx in range(start_idx, end_idx):
        
        parameter = dict(parameters.iloc[training_idx])

        validation_frequency = 1
        collocation_integration_steps = 1
        shuffle = True
        standardize = True
        train = True

        prior_parameters_path = './nn_architecture/parameters.json'
        hybrid_training_folder = "./results/training_abc_smc_2"

        np.random.seed(parameter['seed'])

        if parameter['data'] == 'february_2024':
            data_path = f'./data/february_2024_bis/final/'
        else:
            data_path = f'./data/duffing_oscillator/'

        data_list = []
        for filename in tqdm(os.listdir(f'{data_path}')):
            data_list.append(pd.read_csv(f'{data_path}{filename}'))

        (train_data_list, train_trajectories_idx, 
         val_data_list, val_trajectories_idx, 
         test_data_list, test_trajectories_idx) = split_data(data_list, nb_trajectories, shuffle, train_set_pct, val_set_pct)

        nb_integration_steps = int(parameter['future_temporal_horizon'] / parameter['future_delay'])

        in_variables = system_configuration[parameter['state_configuration']]['state_variable']
        out_variables = system_configuration[parameter['state_configuration']]['state_variable']
        control_variables = system_configuration[parameter['state_configuration']]['control_variable']
        state_variables = system_configuration[parameter['state_configuration']]['state_variable']
            
        if parameter['model'] == 'mlp':
            model = Mlp_narx(
                input_size = len(in_variables) + len(control_variables),
                nb_hidden_layer = parameter['nb_layers'],
                nb_neurones_per_hidden_layer = parameter['nb_neurones_per_layers'],
                output_size = len(in_variables),
                activation = parameter['activation'],
                sequence_duration = parameter['data_dt'],
                dt = parameter['subsampling_dt'],
                delay = parameter['data_dt'] 
            )
            if parameter['transfer_learning']:
                model.load_state_dict(torch.load(f"{training_folder}/best_model_{int(parameter['pretrained_idx'])}.pt"))

        elif parameter['model'] == 'discrepancy':
            
            model = discrepancy(
                input_size = len(in_variables) + len(control_variables),
                nb_hidden_layer = parameter['nb_layers'],
                nb_neurones_per_hidden_layer = parameter['nb_neurones_per_layers'],
                output_size = len(in_variables),
                activation = parameter['activation'],
                sequence_duration = parameter['data_dt'],
                dt = parameter['subsampling_dt'],
                delay = parameter['data_dt'],
                parameter_path = prior_parameters_path)
            
            model.dof2.load_state_dict(torch.load(f"{hybrid_training_folder}/best_model_{int(parameter['pretrained_idx'])}.pt"))
            for param in model.dof2.parameters():
                param.requires_grad = False

        train_in_state, train_in_control, train_out_state = data_preprocessing(
            data_list = train_data_list.copy(),
            data_dt = parameter['data_dt'],
            subsampling_dt = parameter['subsampling_dt'],
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = False,
            smoothing_parameters = smoothing_parameters
        )

        train_in_state_bis, train_in_control_bis, train_out_state_bis, train_out_control_bis = data_preprocessing_(
            data_list = train_data_list.copy(),
            data_dt = parameter['data_dt'],
            subsampling_dt = parameter['subsampling_dt'],
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = False,
            smoothing_parameters = smoothing_parameters
        )
        
        val_in_state, val_in_control, val_out_state = data_preprocessing(
            data_list = val_data_list.copy(),
            data_dt = parameter['data_dt'],
            subsampling_dt = parameter['subsampling_dt'],
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = False,
            smoothing_parameters = smoothing_parameters
        )

        val_in_state_bis, val_in_control_bis, val_out_state_bis, val_out_control_bis = data_preprocessing_(
            data_list = val_data_list.copy(),
            data_dt = parameter['data_dt'],
            subsampling_dt = parameter['subsampling_dt'],
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = False,
            smoothing_parameters = smoothing_parameters
        )

        test_in_state, test_in_control, test_out_state, test_out_control = data_preprocessing_(
            data_list = test_data_list.copy(),
            data_dt = parameter['data_dt'],
            subsampling_dt = parameter['subsampling_dt'],
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = False,
            smoothing_parameters = smoothing_parameters
        )

        reshaped_train_in_state = np.asarray(train_in_state_bis).reshape(-1, np.asarray(train_in_state_bis).shape[2])
        reshaped_train_out_state = np.asarray(train_out_state_bis).reshape(-1, np.asarray(train_out_state_bis).shape[2])
        
        reshaped_val_in_state = np.asarray(val_in_state_bis).reshape(-1, np.asarray(val_in_state_bis).shape[2])
        reshaped_val_out_state = np.asarray(val_out_state_bis).reshape(-1, np.asarray(val_out_state_bis).shape[2])

        reshaped_test_in_state = np.asarray(test_in_state).reshape(-1, np.asarray(test_in_state).shape[2])

        reshaped_train_in_control = np.asarray(train_in_control_bis).reshape(-1, np.asarray(train_in_control_bis).shape[2])
        reshaped_val_in_control = np.asarray(val_in_control_bis).reshape(-1, np.asarray(val_in_control_bis).shape[2])

        std = reshaped_train_in_state.std(axis=0)
        u_std = reshaped_train_in_control.std(axis=0)

        if train:
    
            if parameter['train_on_collocation']:
                
                prior_models = []

                for i in range(15):

                    prior_model = TorchDOF2(
                    parameters_path = prior_parameters_path, 
                    dt = parameter['subsampling_dt'],
                    output_format = 'speed',
                    seed=training_idx)

                    # prior_model = duffing_oscillator(
                    #     parameters_path = prior_parameters_path, 
                    # dt = subsampling_dt,
                    # output_format = 'speed',
                    # seed=training_idx
                    # )

                    prior_model.load_state_dict(
                        torch.load(f'{hybrid_training_folder}/best_model_{i}.pt',  
                                map_location=torch.device(device)
                        )
                    )

                    for param in prior_model.parameters():
                        param.requires_grad = False
                    prior_model.to(device)
                    prior_models.append(prior_model)
                
                data_sample_idx = np.random.randint(0, reshaped_train_in_state.shape[0], int(parameter['nb_collocations']))
                X_col_in = torch.tensor(reshaped_train_in_state[data_sample_idx]).to(device).float() 
                U_col_in = torch.tensor(reshaped_train_in_control[data_sample_idx]).to(device).float() 
                X_col_out = torch.zeros((int(parameter['nb_collocations']), collocation_integration_steps, len(state_variables))).to(device).float()
                U_col_out = torch.zeros((int(parameter['nb_collocations']), collocation_integration_steps, len(control_variables))).to(device).float()
                
                if parameter['ensemble_collocation']:
                    for idx, model_idx in tqdm(enumerate(np.random.randint(0, 15, int(parameter['nb_collocations'])))):
                        pred = X_col_in[idx].unsqueeze(0) + prior_models[model_idx](X_col_in[idx].unsqueeze(0), U_col_in[idx].unsqueeze(0))
                        X_col_out[idx, 0] = pred.detach()
                        U_col_out[idx, 0] = U_col_in[idx].unsqueeze(0).detach()
                else:
                    pred = X_col_in + prior_models[0](X_col_in, U_col_in)
                    X_col_out[:, 0] = pred.detach()
                    U_col_out[:, 0] = U_col_in
                x_col_std = X_col_in.std(axis=0).unsqueeze(0).unsqueeze(0)

                del reshaped_train_in_state, reshaped_val_in_state, prior_models, prior_model
            else: 
                X_col_in = None
                U_col_in = None
                X_col_out = None
                U_col_out = None
                x_col_std = None
            
            std = torch.tensor([[[std[0], std[1]]]]).float().requires_grad_(False).to(device)

            # Instantiate dataloader
            trainDataLoader = SimpleDataLoader(
                train_in_states = train_in_state.copy(),
                train_in_controls = train_in_control.copy(),
                train_out_states = train_out_state.copy(),
                batchsize = parameter['batchsize'],
                past_sequence_duration = 0,
                future_sequence_duration = parameter['future_temporal_horizon'],
                past_delay = 0,
                future_delay = parameter['future_delay'],
                dt = parameter['subsampling_dt'],
                shuffle = shuffle,
                device = device
            )

            valDataLoader = SimpleDataLoader(
                train_in_states = val_in_state.copy(),
                train_in_controls = val_in_control.copy(),
                train_out_states = val_out_state.copy(),  
                batchsize = parameter['batchsize'],
                past_sequence_duration = 0,
                future_sequence_duration = parameter['future_temporal_horizon'],
                past_delay = 0,
                future_delay = parameter['future_delay'],
                dt = parameter['subsampling_dt'],
                shuffle = shuffle,
                device = device
            )

            training_config = {
                'in_states_variables' : list(in_variables),
                'out_states_variables': list(out_variables),
                'states_variables': list(state_variables),
                'control_variables':list(control_variables),
                'training_idx' : [int(idx) for idx in train_trajectories_idx],
                'validation_idx': [int(idx) for idx in val_trajectories_idx],
                'test_idx' : [int(idx) for idx in test_trajectories_idx],
            }

            save_to_json(
                training_config,
                f'{training_folder}/training_config_{training_idx}.json'
            )

            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr = parameter['learning_rate'])
            best_loss = np.inf

            training_results = {}
            training_results["train_loss"] = []
            training_results["physics_loss"] = []
            training_results["val_loss"] = []

            save_to_json(
                training_results,
                f'{training_folder}/training_results_{training_idx}.json'
                )
                    
            train_loss_list = []
            val_loss_list = []
            for epochs in range(parameter['nb_epochs']):
        
                train_loss, physics_loss = train_loop(
                    model,
                    optimizer,
                    trainDataLoader,
                    standardize,
                    std,
                    x_col_std,
                    nb_integration_steps,
                    X_col_out = X_col_out,
                )

                if epochs % validation_frequency == 0:

                    val_loss = validation_loop(
                        model,
                        valDataLoader,
                        standardize = standardize,
                        std = std,
                        nb_integration_steps = nb_integration_steps,
                    )

                    print(f'Epoch : {epochs} | train loss : {np.mean(train_loss):.4f} | physics loss : {np.mean(physics_loss):.4f} | val loss : {np.mean(val_loss):.4f}')
                    print()

                    if parameter['early_stopping']:
                        if np.mean(val_loss) < best_loss:
                            best_loss = np.mean(val_loss)  
                            torch.save(model.state_dict(), f'{training_folder}/best_model_{training_idx}.pt')
                            saved_idx = epochs
                            print("Save model")
                    else:
                        torch.save(model.state_dict(), f'{training_folder}/best_model_{training_idx}.pt')

                    training_results["train_loss"].append(float(np.mean(train_loss)))
                    training_results["val_loss"].append(float(np.mean(val_loss)))
        
            save_to_json(
                    training_results,
                    f'{training_folder}/training_results_{training_idx}.json'
                    )
                
            del trainDataLoader, valDataLoader
            del train_in_state, train_in_control, train_out_state,
            del train_data_list, val_data_list, test_data_list
            del X_col_in, X_col_out, U_col_in, U_col_out
            del val_in_state, val_in_control, val_out_state
            del data_list

            gc.collect()
            torch.cuda.empty_cache()
        
        test = True

        if test:
            
            if train == False:
                std = torch.tensor([[[std[0], std[1]]]]).float().requires_grad_(False).to(device)

            # Evaluate on train set
            model.load_state_dict(torch.load(f'{training_folder}/best_model_{training_idx}.pt'))
            model.to(device)
            model.eval()

            nb_test_trajectories = len(test_in_state)
            sequence_duration = 10
            s = 0
            nb_step = int(len_trajectories / parameter['subsampling_dt']) - 1 - s
            
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
            for element in list(state_variables)+['rmse']:
                for time in times:
                    rmse[f'{element}_{time}'] = []

            for time in times:
                idx_limit = int(time/10 * X_predictions_tensor.shape[1]+1)
                RMSE = torch.mean(torch.sqrt(torch.mean(((X_predictions_tensor[:, :idx_limit] - X_out_tensor[:, :idx_limit]))**2, axis=1)), axis=0)
                for idx, element in enumerate(list(state_variables)):
                    rmse[f'{element}_{time}'] = RMSE[idx].item()
                rmse[f'rmse_{time}'] = RMSE.mean().item()

            create_folder_if_not_exists(f'{training_folder}/results/')
            save_to_json(
                    rmse,
                    f'{training_folder}/results/rmse_{training_idx}.json'
                )
            
            # create_folder_if_not_exists(f'{training_folder}/figures/')
            # plt.figure(figsize = (10, 10))
            # plt.plot(np.arange(0, X_predictions_tensor.shape[1], 1) * 0.05, X_predictions_tensor[0, :, 1].detach().cpu().tolist())
            # plt.plot(np.arange(0, X_predictions_tensor.shape[1], 1) * 0.05, X_out_tensor[0, :, 1].detach().cpu().tolist())
            # plt.xlabel('time')
            # plt.ylabel('dpsidt (rad/s)')
            # plt.grid()
            # plt.savefig(f'{training_folder}/figures/dpsidt_{training_idx}.png')
            # plt.close()

            # plt.figure(figsize = (10, 10))
            # plt.plot(np.arange(0, X_predictions_tensor.shape[1], 1) * 0.05, X_predictions_tensor[0, :, 0].detach().cpu().tolist())
            # plt.plot(np.arange(0, X_predictions_tensor.shape[1], 1) * 0.05, X_out_tensor[0, :, 0].detach().cpu().tolist())
            # plt.xlabel('time')
            # plt.ylabel('vy (m/s)')
            # plt.grid()
            # plt.savefig(f'{training_folder}/figures/dvydt_{training_idx}.png')
            # plt.close()

            # plt.close('all')

            del model  
            del X_in_tensor, U_in_tensor, X_out_tensor, X_predictions_tensor
            del test_in_state, test_in_control, test_out_state   