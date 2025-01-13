# example of nnmd package usage

import os
import shutil

import torch

from nnmd.nn import HDNN
from nnmd.nn.dataset import make_atomic_dataset
from nnmd.util import input_parser, train_val_test_split
from nnmd.features import params_for_G2, params_for_G4

from torch.utils.data.dataset import TensorDataset

device = torch.device('cuda')
dtype = torch.float32

print("Get info from traj simulation: ", end = '')
input_data = input_parser("input/input.yaml")

# inputs
cartesians = torch.tensor(input_data['atomic_data']['reference_data']['cartesians'], dtype = dtype, device = device)
cartesians.requires_grad = True
n_structs = cartesians.size(0)
n_atoms = cartesians.size(1)

# targets
energies = torch.tensor(input_data['atomic_data']['reference_data']['energies'], dtype = dtype, device = device)
forces = torch.tensor(input_data['atomic_data']['reference_data']['forces'], dtype = dtype, device = device)

# for automatic params computation
# must be defined cutoffs and number of radial and angular functions
r_cutoff_g2 = 4.0
r_cutoff_g4 = 4.0
n_radial = 4
n_angular = 12

features = [2] * n_radial + [4] * n_angular
params = params_for_G2(n_radial, r_cutoff_g2) + params_for_G4(n_angular, r_cutoff_g4)
symm_funcs_data = {'features': features, 'params': params}

print("done")

print(f"Separate data to train and test datasets:", sep = '\n', end = ' ')
# ~80% - train, ~10% - test and validation
train_val_test_ratio = (0.8, 0.1, 0.1)
train_dataset, val_dataset, test_dataset = train_val_test_split(TensorDataset(cartesians, energies, forces), train_val_test_ratio)

# convert train data to atomic dataset with symmetric functions
train_dataset = make_atomic_dataset(train_dataset, symm_funcs_data, saved = True, train = True, path = "input/train")
val_dataset = make_atomic_dataset(val_dataset, symm_funcs_data, saved = True, train = True, path = "input/val")
test_dataset = make_atomic_dataset(test_dataset, symm_funcs_data, saved = True, train = True, path = "input/test")
print("done")

# train model
train = input_data['neural_network']['train']

# save model params as files in <path> directory
save = input_data['neural_network']['save']
# path to save model
path = input_data['neural_network']['path']

# Atomic NN input_size in hidden layers
input_size = n_radial + n_angular

print("Create an instance of NN and config its subnets:", end = ' ')

net = HDNN(dtype = dtype)
net.config(input_data['neural_network'], input_size)

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