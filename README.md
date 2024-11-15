# JIS code

This repository contains the implementation of algorithms and experiments for the JIS journal. The code is structured to facilitate the execution of the ABC-SMC algorithm, model training, and performance visualization.

## 1) Installation

Before running the code, make sure to install all required Python packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

## 2) Run ABC-SMC algorithm for each time integration steps

```bash
python abc_smc_train.py --training_name abc_smc_0 --training_parameters abc_smc_parameters_0
python abc_smc_train.py --training_name abc_smc_1 --training_parameters abc_smc_parameters_1
python abc_smc_train.py --training_name abc_smc_2 --training_parameters abc_smc_parameters_2
```

## 3) Pre-train model on data simulated from bicycle model calibrated by ABC-SMC algorithm

```bash
python train.py --training_name train_results --training_parameters training_parameters --start_index 0 --end_index 45
```

## 4) Run all trainings containing each compared methods (Data-based, discrepancy and transfer-learning methods) :

```bash
python train.py --training_name train_results --training_parameters training_parameters --start_index 45 --end_index 450
```

## 5) Visualize performances by running test_results.ipynb
