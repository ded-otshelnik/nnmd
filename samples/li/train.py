# example of nnmd package usage

import os
import shutil

import torch

from nnmd.nn import HDNN
from nnmd.nn.dataset import make_atomic_dataset
from nnmd.features.auto_params import params_for_G2, params_for_G4
from nnmd.util import train_val_test_split
from nnmd.io import input_parser

import warnings
warnings.filterwarnings("ignore")
dtype = torch.float32

input_data = input_parser("input/input.yaml")
n_atoms = input_data['atomic_data']['n_atoms']

# parameters for symmetry functions    
r_cutoff = 3.54
    
params_g2 = [
            [r_cutoff, 0.001, 0.0],
            [r_cutoff, 0.01, 0.0],
            [r_cutoff, 0.03, 0.0],
            [r_cutoff, 0.05, 0.0],
            [r_cutoff, 0.7, 0.0],
            [r_cutoff, 0.1, 0.0],
            [r_cutoff, 0.2, 0.0],
            [r_cutoff, 0.4, 0.0],
            [r_cutoff, 0.5, 0.0],
            [r_cutoff, 0.7, 0.0],
            [r_cutoff, 0.9, 0.0],
            [r_cutoff, 1.0, 0.0],
    ]
params_g4 = [
            [r_cutoff, 0.01, 3, -1],
            [r_cutoff, 0.02, 3, 1]
    ]

n_radial = len(params_g2)
n_angular = len(params_g4)
features = [2] * n_radial + [4] * n_angular
params = params_g2 + params_g4
symm_funcs_data = {'Li': {'features': features, 'params': params}}
# for feature, params in zip(symm_funcs_data['Li']['features'], symm_funcs_data['Li']['params']):
#     print(f"G{feature}", params, sep = ": ")
print("Get info from traj simulation: done")

# convert train data to atomic dataset with symmetry functions
dataset = make_atomic_dataset(input_data['atomic_data'], symm_funcs_data)

# ~80% - train, ~10% - test and validation
train_val_test_ratio = (0.8, 0.1, 0.1)
train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, train_val_test_ratio)
print(f"Separate data to train and test datasets: done")

# train model
train = input_data['neural_network']['train']
# save model params as files in <path> directory
save = input_data['neural_network']['save']
# path to save model
path = input_data['neural_network']['path']

# Atomic NN input_size in hidden layers
input_sizes = [n_radial + n_angular]
output_sizes = [1]

print("Create an instance of NN and config its subnets:", end = ' ')

net = HDNN(dtype = dtype)
net.config(input_data['neural_network'], input_sizes, output_sizes)

print("done")

try:
    if train:
        print("Training:", end = ' ')
        net.fit(train_dataset, val_dataset, test_dataset)
        
    if save:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors = True)
        os.mkdir(path)

        net.save_model(path)
        print("Saving model: done")
        
except KeyboardInterrupt:
    if os.path.exists("checkpoint"):
        shutil.rmtree("checkpoint", ignore_errors = True)
    os.mkdir("checkpoint")

    net.save_model("checkpoint")
    print("Training is stopped. Model is saved")