#!/usr/bin/env python
# -*- coding: utf-8 -*-

# example of nnmd package usage with gpaw simulation

import os
import random
import shutil
import time 
import argparse

from nnmd.nn import dataset

import torch

import nnmd
from nnmd.nn import Neural_Network
from nnmd.nn.dataset import make_atomic_dataset
from nnmd.util import gpaw_parser

params_parser = argparse.ArgumentParser(description = "Sample code of nnmd usage")
params_parser.add_argument("gpaw_file", help = "File of GPAW simulation")
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

print("Get info from GPAW simulation: ", end = '')
n_structs, n_atoms, cartesians, f_dft, e_dft = gpaw_parser(args.gpaw_file)
print("done")

device = torch.device('cuda') if use_cuda else torch.device('cpu')
dtype = torch.float32

print(f"Separate data to train and test datasets:", sep = '\n', end = ' ')

# params of symmetric functions
symm_func_params = {"r_cutoff": 6.0,
                    "eta": 0.01,
                    "k": 1,
                    "rs": 0.5,
                    "lambda": -1,
                    "xi": 3}
h = 0.01

# shuffle data
data = list(zip(cartesians, e_dft, f_dft))
random.shuffle(data)
cartesians, e_dft, f_dft = zip(*data)

# ~80% - train, ~20% - test
sep = int(0.8 * n_structs)

# inputs and targets
train_cartesians, test_cartesians = torch.as_tensor(cartesians[:sep], device = device, dtype = dtype), \
                                    torch.as_tensor(cartesians[sep:],  device = device, dtype = dtype)
train_e_dft, test_e_dft = torch.as_tensor(e_dft[:sep], device = device, dtype = dtype), \
                          torch.as_tensor(e_dft[sep:], device = device, dtype = dtype)
train_f_dft, test_f_dft = torch.as_tensor(f_dft[:sep], device = device, dtype = dtype), \
                          torch.as_tensor(f_dft[sep:], device = device, dtype = dtype)
train_dataset = make_atomic_dataset(train_cartesians, symm_func_params, h, device,
                                    train_e_dft, train_f_dft, train = True)
test_dataset = make_atomic_dataset(test_cartesians, symm_func_params, h, device) 
print("done")

# params that define what NN will do 
# load pre-trained models
load_models = False
path = 'models_new'
# train model
train = True
# save model params as files in <path> directory
save = True
# test model
test = True

# Atomic NN nodes in hidden layers
input_nodes = 5
hidden_nodes = [30, 30]
mu = 1
learning_rate = 0.1

# sigma = 0.1
# for g in train_dataset.g:
#     print(nnmd.nnmd_cpp.cuda.calculate_gdf(g, len(g), len(g[0]), input_nodes, sigma))

print("Create an instance of NN and config its subnets:", end = ' ')
net = Neural_Network()
net.config(hidden_nodes = hidden_nodes, 
           use_cuda = use_cuda,
           dtype = dtype,
           n_atoms = n_atoms,
           input_nodes = input_nodes,
           load_models = load_models,
           path = path,
           mu = mu,
           learning_rate = learning_rate)
print("done")

if train:
    # parameters of training 
    batch_size = 16
    epochs = 100

    print(f"Training sample size: {len(train_dataset)}", "Training:", sep = '\n') 

    start = time.time()
    net.fit(train_dataset, batch_size, epochs)
    end = time.time()

    train_time = end - start
    print(f"Training time ({'GPU' if device.type == 'cuda' else 'CPU'}): {train_time:.3f} s")

if save:
    print("Saving model: ", end = '')

    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors = True)
    os.mkdir(path)

    net.save_model(path)
    print("done")

if test:
    print("Testing:", end = ' ')

    start = time.time()
    test_loss, test_e_loss, test_f_loss = 0, 0, 0
    for cartesian, e_dft, f_dft in zip(test_cartesians, test_e_dft, test_f_dft):
        test_e_nn, test_f_nn = net.predict(cartesian, symm_func_params, h)
        loss = net.loss(test_e_nn.unsqueeze(0), e_dft, test_f_nn, f_dft)
        test_loss += loss[0]
        test_e_loss += loss[1]
        test_f_loss+= loss[2]
    end = time.time()

    print("done", f"Testing sample size: {len(test_dataset)} ",
          f"Testing time ({'GPU' if device.type == 'cuda' else 'CPU'}): {(end - start):.3f} s",
          sep = '\n')

    test_e_loss = test_e_loss.cpu().detach().numpy()
    test_f_loss = test_f_loss.cpu().detach().numpy()
    test_loss = test_loss.cpu().detach().numpy()

    print(f"test: RMSE E = {test_e_loss:.4f}, RMSE F = {test_f_loss:.4f}, RMSE total = {test_loss:.4f} eV")