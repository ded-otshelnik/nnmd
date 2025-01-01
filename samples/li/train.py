# example of nnmd package usage with gpaw simulation

import os
import shutil
import time 
import argparse

import numpy as np

import torch

from nnmd.nn import HDNN
from nnmd.nn.dataset import make_atomic_dataset
from nnmd.util import traj_parser, train_val_test_split

from torch.utils.data.dataset import TensorDataset

params_parser = argparse.ArgumentParser(description = "Sample code of nnmd usage")
params_parser.add_argument("data_file", type = str, help = "Path to file with atomic data")
args = params_parser.parse_args()

device = torch.device('cuda')
dtype = torch.float32

print("Get info from traj simulation: ", end = '')
cartesians, forces, energies, velocities = traj_parser(args.data_file)

n_structs = len(cartesians)
n_atoms = len(cartesians[0])

# inputs
cartesians = torch.as_tensor(cartesians, dtype = dtype, device = device)
# targets
energies = torch.tensor(energies, dtype = dtype, device = device)
forces = torch.tensor(forces, dtype = dtype, device = device)

print("done")

print(f"Separate data to train and test datasets:", sep = '\n', end = ' ')

# params of symmetric functions
# euclidean distances between atoms in the first structure
distances = torch.norm(cartesians[0, :, None, :] - cartesians[0, None, :, :], dim = -1, p = 2)
# all distances that are 0 will be replaced with inf so that they do not affect the minimum
min_distance = torch.where(distances == 0, torch.as_tensor(np.inf, device = device), distances).min().item()

# params of symmetric functions
symm_func_params = {"r_cutoff": 7.0,
                    "eta": -2,
                    "rs": 3,
                    "kappa": 2,
                    "lambda": 1,
                    "zeta": 4,
                    "h": 0.1}
# ~80% - train, ~10% - test and validation
train_val_test_ratio = (0.8, 0.1, 0.1)
train_dataset, val_dataset, test_dataset = train_val_test_split(TensorDataset(cartesians, energies, forces), train_val_test_ratio)


# convert train data to atomic dataset with symmetric functions
train_dataset = make_atomic_dataset(train_dataset, symm_func_params, device, train = True, path = "train")
val_dataset = make_atomic_dataset(val_dataset, symm_func_params, device, train = True, path = "val")
test_dataset = make_atomic_dataset(test_dataset, symm_func_params, device)
print("done")

torch.cuda.empty_cache()

# params that define what NN will do 
# load pre-trained models
load_models = False
path = 'model'
# train model
train = True
# save model params as files in <path> directory
save = True
# test model
test = True

# Atomic NN nodes in hidden layers
input_nodes = 5
hidden_nodes = [64, 32, 8]
mu = 1
learning_rate = 0.01

print("Create an instance of NN and config its subnets:", end = ' ')
net = HDNN()
net.config(hidden_nodes = hidden_nodes, 
           use_cuda = True,
           dtype = dtype,
           n_atoms = n_atoms,
           input_nodes = input_nodes,
           load_models = load_models,
           path = path + "/atomic_nn_Li.pt",
           mu = mu,
           learning_rate = learning_rate)
print("done")

try:
    if train:
        batch_size = 1000
        epochs = 1

        start = time.time()
        net.fit(train_dataset, val_dataset, batch_size, epochs)
        end = time.time()

        train_time = end - start
        net.time_log.info(f"Training time ({'GPU' if device.type == 'cuda' else 'CPU'}): {train_time:.3f} s")
        
    if save:
        print("Saving model: ", end = '')

        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors = True)
        os.mkdir(path)

        net.save_model(path + "/atomic_nn_Li.pt")
        print("done")

    if test:
        print("Testing:", end = ' ')
        start = time.time()
        test_loss, test_e_loss, test_f_loss = 0, 0, 0
        for cartesian, energy_struct, force_struct in test_dataset:
            test_e_nn, test_f_nn = net.predict(cartesian, symm_func_params)
            loss = net.loss(test_e_nn.sum(dim = 0), energy_struct.unsqueeze(0), test_f_nn, force_struct)
            test_loss   += loss[0]
            test_e_loss += loss[1]
            test_f_loss += loss[2]
        end = time.time()

        print("done")

        test_e_loss = test_e_loss.cpu().detach().numpy() / len(test_dataset)
        test_f_loss = test_f_loss.cpu().detach().numpy() / len(test_dataset)
        test_loss = test_loss.cpu().detach().numpy() / len(test_dataset)

        net.net_log.info(f"Testing sample size: {len(test_dataset)}")
        net.net_log.info(f"Testing: RMSE E = {test_e_loss:e}, RMSE F = {test_f_loss:e}, RMSE total = {test_loss:e} eV")
        
except:
    print("Training is stopped. Model is saved")
    net.save_model("checkpoint.pt")