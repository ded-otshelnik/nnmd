#!/usr/bin/env python
# -*- coding: utf-8 -*-

# example of nnmd package usage with gpaw simulation

import os
import shutil
import time 
import argparse

import torch
from sklearn.model_selection import train_test_split

from nnmd.nn import Neural_Network
from nnmd.nn.dataset import make_atomic_dataset
from nnmd.util import gpaw_parser

params_parser = argparse.ArgumentParser(description = "Sample code of nnmd usage")
params_parser.add_argument("gpaw_file", help = "File of GPAW simulation")
params_parser.add_argument("-c", "--use_cuda",
                            action = argparse.BooleanOptionalAction,
                            help = "Enable/disable CUDA usage.")
args = params_parser.parse_args()

# check if GPU usage is needed and available 
use_cuda = (args.use_cuda is not None) and torch.cuda.is_available()

# print GPU availability status
if use_cuda:
    print("GPU usage is enabled")
elif args.use_cuda and not torch.cuda.is_available():
    print("Pytorch compiled/downloaded without CUDA support. GPU is disabled")
else:
    print("GPU usage is disabled")

print("Get info from GPAW simulation: ", end = '')
n_structs, n_atoms, cartesians, f_dft, e_dft = gpaw_parser(args.gpaw_file)
print("done")

print("Create an instance of NN:", end = ' ')
# Global parameters of HDNN 
epochs = 1
batch_size = 16
# Atomic NN, nodes amount in hidden layers
hidden_nodes = [16, 8]
# create an instance
net = Neural_Network(hidden_nodes = hidden_nodes,
                     epochs = epochs,
                     use_cuda = use_cuda)
print("done")

device = torch.device('cuda') if use_cuda else torch.device('cpu')
dtype = torch.float32

print(f"Move NN to {'GPU' if device.type == 'cuda' else 'CPU'}:", end = ' ')
# move NN to right device 
net.to(device = device)
print("done")

print(f"Separate data to train and test datasets:", end = ' ')

# params of symmetric functions
rc = 12.0
eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3

# ~80% - train, ~20% - test
sep = int(0.8 * n_structs)
train_cartesians, test_cartesians = torch.as_tensor(cartesians[:sep], device = device, dtype = dtype), \
                                    torch.as_tensor(cartesians[sep:-1],  device = device, dtype = dtype)
train_e_dft, test_e_dft = torch.as_tensor(e_dft[:sep], device = device, dtype = dtype), \
                          torch.as_tensor(e_dft[sep:], device = device, dtype = dtype)
train_f_dft, test_f_dft = torch.as_tensor(f_dft[:sep], device = device, dtype = dtype), \
                          torch.as_tensor(f_dft[sep:], device = device, dtype = dtype)

train_dataset = make_atomic_dataset(train_cartesians, rc, eta, rs, k, _lambda, xi, train_e_dft, train_f_dft, device)
train_dataset.g.requires_grad = True
test_dataset = make_atomic_dataset(test_cartesians, rc, eta, rs, k, _lambda, xi, test_e_dft, test_f_dft, device)
print("done")
# params that define what NN will do 
load_models = True
# train model
train = False
# save model params as files in <path> directory
save = False
# test model
test = True
path = 'models'

print("Config subnets:", end = ' ')

net.config(n_atoms, load_models = load_models, path = path)

print("done")

if train:

    print("Training:")
    print(f"\nTraining sample size: {len(train_dataset)} ") 

    start = time.time()
    net.fit(train_dataset, batch_size)
    end = time.time()

    train_time = (end - start)
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
    test_e_nn, test_f_nn = net.predict(test_dataset)
    end = time.time()

    print("done")
    print(f"\nTesting sample size: {len(test_dataset)} ")
    print(f"Testing time ({'GPU' if device.type == 'cuda' else 'CPU'}): {(end - start):.3f} s")

    test_loss, test_e_loss, test_f_loss = net.loss(test_e_nn, test_e_dft, test_f_nn, test_f_dft)

    test_e_loss = test_e_loss.cpu().detach().numpy()
    test_f_loss = test_f_loss.cpu().detach().numpy()
    test_loss = test_loss.cpu().detach().numpy()

    test_loss_info = f"test: RMSE E = {test_e_loss:.4f}, RMSE F = {test_f_loss:.4f}, RMSE total = {test_loss:.4f} eV"
    print(test_loss_info)