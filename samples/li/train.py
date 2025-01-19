# example of nnmd package usage

import os
import shutil

import torch

from nnmd.nn import HDNN
from nnmd.nn.dataset import make_atomic_dataset
from nnmd.util import train_val_test_split
from nnmd.io import input_parser

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda')
dtype = torch.float32

print("Get info from traj simulation:", end = ' ')
input_data = input_parser("input/input.yaml")
n_atoms = input_data['atomic_data']['n_atoms']

# for automatic params computation
# must be defined cutoffs and number of radial and angular functions
r_cutoff = 3.54
n_radial = 4
n_angular = 4
features = [2] * n_radial + [4] * n_angular
params_g2 = [
    [r_cutoff, 2, 0.1],
    [r_cutoff, 2, 0.3],
    [r_cutoff, 2, 0.5],
    [r_cutoff, 2, 0.7]
]
params_g4 = [
    [r_cutoff, 1.0, 3, -1],
    [r_cutoff, 1.0, 3, 1],
    [r_cutoff, 2.0, 3, -1],
    [r_cutoff, 2.0, 3, 1]
]

params = params_g2 + params_g4
symm_funcs_data = {'Li': {'features': features, 'params': params}}

print("done")

# convert train data to atomic dataset with symmetric functions
saved = False
for atom_spec in input_data['neural_network']['atom_species']:
    if os.path.exists(f"dg_{atom_spec}.pt"):
        saved = True
    else:
        saved = False
        print(f"Symmetry functions for {atom_spec} are not calculated yet, calculating...")
        break
    print(f"Symmetry functions for {atom_spec} are already calculated")
    
dataset = make_atomic_dataset(input_data['atomic_data'], symm_funcs_data, device, saved = saved)
print(f"Separate data to train and test datasets: done")

# ~80% - train, ~10% - test and validation
train_val_test_ratio = (0.8, 0.1, 0.1)
train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, train_val_test_ratio)
print("done")

# train model
train = input_data['neural_network']['train']

# save model params as files in <path> directory
save = input_data['neural_network']['save']
# path to save model
path = input_data['neural_network']['path']

# Atomic NN input_size in hidden layers
input_sizes = [n_radial + n_angular]
output_sizes = n_atoms

print("Create an instance of NN and config its subnets:", end = ' ')

net = HDNN(dtype = dtype)
net.config(input_data['neural_network'], input_sizes, output_sizes)

print("done")

try:
    if train:
        print("Training:", end = ' ')
        net.fit(train_dataset, val_dataset, test_dataset)
        
    if save:
        print("Saving model: ", end = '')

        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors = True)
        os.mkdir(path)

        net.save_model(path)
        print("done")
        
except KeyboardInterrupt:
    print("Training is stopped. Model is saved")
    
    if os.path.exists("checkpoint"):
        shutil.rmtree("checkpoint", ignore_errors = True)
    os.mkdir("checkpoint")
    net.save_model("checkpoint")