#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

# import warnings
# warnings.filterwarnings('ignore')

params_parser = argparse.ArgumentParser(description = "Sample code of nnmd usage")
params_parser.add_argument("data_file", type = str, help = "Path to file with atomic data")
params_parser.add_argument("-c", "--use_cuda",
                            action = argparse.BooleanOptionalAction,
                            help = "Enable/disable CUDA usage.")
args = params_parser.parse_args()

use_cuda = (args.use_cuda is not None) and torch.cuda.is_available()
if use_cuda:
    print("GPU usage is enabled")
elif args.use_cuda and not torch.cuda.is_available():
    print("Pytorch compiled/downloaded without CUDA support. GPU is disabled")
else:
    print("GPU usage is disabled")

device = torch.device('cuda') if use_cuda else torch.device('cpu')
dtype = torch.float32

print("Get info from traj simulation: ", end = '')
# cartesians, forces, energies, velocities = traj_parser(args.data_file)
cartesians = np.load("cartesians_actual.npy.npz")['cartesians'].tolist()
energies = np.load("energies_actual.npy.npz")['energies'].tolist()
forces = np.load("forces_actual.npy.npz")['forces'].tolist()
print("done")

print(f"Separate data to train and test datasets:", sep = '\n', end = ' ')

# params of symmetric functions
symm_func_params = {"r_cutoff": 6.0,
                    "eta": 0.01,
                    "k": 1,
                    "rs": 0.5,
                    "lambda": -1,
                    "xi": 3}
h = 0.01

n_structs = len(cartesians)
n_atoms = len(cartesians[0])

# inputs
cartesians = torch.as_tensor(cartesians, dtype = dtype, device = device)
# targets
energies = torch.tensor(energies, dtype = dtype, device = device)
forces = torch.tensor(forces, dtype = dtype, device = device)


# ~80% - train, ~10% - test and validation
train_val_test_ratio = (0.8, 0.1, 0.1)
train_dataset, val_dataset, test_dataset = train_val_test_split(TensorDataset(cartesians, energies, forces), train_val_test_ratio)

# convert train data to atomic dataset with symmetric functions
train_dataset = make_atomic_dataset(train_dataset, symm_func_params, h, device, train = True)
val_dataset = make_atomic_dataset(val_dataset, symm_func_params, h, device, train = True)
test_dataset = make_atomic_dataset(test_dataset, symm_func_params, h, device)

print("done")

# params that define what NN will do 
# load pre-trained models
load_models = False
path = 'models'
# train model
train = True
# save model params as files in <path> directory
save = False
# test model
test = True

# Atomic NN nodes in hidden layers
input_nodes = 5
hidden_nodes = [30, 30]
mu = 1
learning_rate = 0.1

print("Create an instance of NN and config its subnets:", end = ' ')
net = HDNN()
net.config(hidden_nodes = hidden_nodes, 
           use_cuda = use_cuda,
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
        batch_size = 100
        epochs = 20

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
            test_e_nn, test_f_nn = net.predict(cartesian, symm_func_params, h)
            loss = net.loss(test_e_nn.sum(dim = 0), energy_struct.unsqueeze(0), test_f_nn, force_struct)
            test_loss += loss[0]
            test_e_loss += loss[1]
            test_f_loss+= loss[2]
        end = time.time()

        print("done")
        net.time_log.info(f"Testing time ({'GPU' if device.type == 'cuda' else 'CPU'}): {(end - start):.3f} s")

        test_e_loss = test_e_loss.cpu().detach().numpy() / len(test_dataset)
        test_f_loss = test_f_loss.cpu().detach().numpy() / len(test_dataset)
        test_loss = test_loss.cpu().detach().numpy() / len(test_dataset)

        net.net_log.info(f"Testing sample size: {len(test_dataset)}")
        net.net_log.info(f"Testing: RMSE E = {test_e_loss:e}, RMSE F = {test_f_loss:e}, RMSE total = {test_loss:e} eV")
        
except KeyboardInterrupt:
    print("Training is stopped. Model is saved")
    net.save_model("checkpoint.pt")