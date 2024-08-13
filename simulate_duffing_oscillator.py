import sys
sys.path.append('./')

from system import duffing_oscillator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import create_folder_if_not_exists

if __name__ == '__main__':
    
    delta_t = 0.001
    T = 10
    nb_integration_step = int(T / delta_t)
    nb_simulation = 100
    seed = 42

    intial_conditions = (np.random.rand(nb_simulation, 2) - 0.5) *  1
    
    folder_path = f'./data/duffing_oscillator/'
    create_folder_if_not_exists(folder_path)

    for n in range(nb_simulation):

        t = 0

        results = {'x' : [], 'y':[], 'x_' : [], 'y_':[],'f':[]}

        X = intial_conditions[n]
        X_ = intial_conditions[n]

        for i in range(nb_integration_step):
            
            f = np.random.rand() * np.cos((1+np.random.rand())*t)

            results['x'].append(X[0] + np.random.randn() * 0.1)
            results['y'].append(X[1] + np.random.randn() * 0.1)
            results['f'].append(f + np.random.randn() * 0.01)

            Xdt = duffing_oscillator(X, f, beta=1)
            X = np.asarray(Xdt) * delta_t + X

            results['x_'].append(X_[0] + np.random.randn() * 0.1)
            results['y_'].append(X_[1] + np.random.randn() * 0.1)

            Xdt_ = duffing_oscillator(X_, f, discrepancy=True)
            X_ = np.asarray(Xdt_) * delta_t + X_

            t += delta_t

        data = pd.DataFrame(results)
        data.to_csv(f'{folder_path}/{n}.csv')