import torch
from torch import Tensor
from collections import OrderedDict
from .vehicle_architecture import TorchDOF2, duffing_oscillator
import numpy as np
import json

class Mlp(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network model.

    Args:
        input_size (int): The size of the input feature.
        nb_hidden_layer (int): The number of hidden layers.
        nb_neurons_per_hidden_layer (int): The number of neurons in each hidden layer.
        output_size (int): The size of the output layer.
        activation (str): The activation function to be used ('relu', 'elu', 'sigmoid', 'tanh', 'softplus').

    Attributes:
        activation (torch.nn.Module): The activation function used in the hidden layers.
        layers (torch.nn.Sequential): The sequence of layers in the MLP.

    Example:
    ```
    mlp = MLP(input_size=64, nb_hidden_layer=2, nb_neurons_per_hidden_layer=128, output_size=10, activation='relu')
    output = mlp(input_data)
    ```

    """

    def __init__(
            self,
            input_size: int,
            nb_hidden_layer: int,
            nb_neurons_per_hidden_layer: int,
            output_size: int,
            activation: str,
        ):

        super(Mlp, self).__init__()

        layers = [input_size] + [nb_neurons_per_hidden_layer] * nb_hidden_layer + [output_size]
    
        # set up layer order dict
        if activation == 'relu':
            self.activation = torch.nn.ReLU
        if activation == 'elu':
            self.activation = torch.nn.ELU
        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'softplus':
            self.activation = torch.nn.Softplus

        depth = len(layers) - 1
        layer_list = list()
        for i in range(depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, X):
        return self.layers(X)
    
class Nn_base(torch.nn.Module):

    def __init__(
            self,
            input_size,
            output_size,
            delay,
            sequence_duration,
            dt,
        ):

        super(Nn_base, self).__init__()

        self.register_buffer('input_size', torch.tensor(input_size))
        self.register_buffer('output_size', torch.tensor(output_size))
        self.register_buffer('delay', torch.tensor(delay).float())
        self.register_buffer('sequence_duration', torch.tensor(sequence_duration).float())
        self.register_buffer('dt', torch.tensor([dt]).float())
        self.register_buffer('z_scaling', torch.tensor(False))
        self.register_buffer('std', torch.tensor(torch.tensor([[1] * self.output_size]).float()))
        self.register_buffer('mean', torch.tensor([[0] * self.output_size]).float())

    def set_z_scale(self, z_scale):
        self.register_buffer('z_scaling', torch.tensor(z_scale))
    
    def set_std(self, std):
        self.register_buffer('std', torch.tensor([std]).float())

    def set_mean(self, mean):
        self.register_buffer('mean', torch.tensor([mean]).float())

    def set_delay(self, delay):
        self.register_buffer('delay', torch.tensor(delay).float())

    def set_sequence_duration(self, sequence_duration):
        self.register_buffer('sequence_duration', torch.tensor(sequence_duration).float())

    def set_dt(self, dt):
        self.register_buffer('dt', torch.tensor([dt]).float())

class Mlp_narx(Nn_base):

    def __init__(
            self, 
            input_size,
            nb_hidden_layer,
            nb_neurones_per_hidden_layer,
            output_size,
            activation,
            dt,
            sequence_duration,
            delay,    
        ):

        super(Mlp_narx, self).__init__(
            input_size,
            output_size,
            delay,
            sequence_duration,
            dt,
        )

        self.fc = Mlp(
            input_size,
            nb_hidden_layer,
            nb_neurones_per_hidden_layer,
            output_size,
            activation,
        )
        
    def forward(self, X, U):
        out = self.fc(torch.concat((X, U), dim=1))
        return out

class discrepancy(Nn_base):

    def __init__(
            self, 
            input_size,
            nb_hidden_layer,
            nb_neurones_per_hidden_layer,
            output_size,
            activation,
            dt,
            sequence_duration,
            delay,
            parameter_path
        ):

        super(discrepancy, self).__init__(
            input_size,
            output_size,
            delay,
            sequence_duration,
            dt,
        )

        self.dof2 = TorchDOF2(
            parameter_path, 
            dt, 
            output_format='speed'
        )

        self.fc = Mlp(
            input_size,
            nb_hidden_layer,
            nb_neurones_per_hidden_layer,
            output_size,
            activation,
        )
        
    def forward(self, X, U):
        out1 = self.dof2(X, U)
        out2 = self.fc(torch.concat((X, U), dim=1))
        return out1 + out2