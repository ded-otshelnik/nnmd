#!/usr/bin/env python
# -*- coding: utf-8 -*-

# test pre-trained neural network on gpaw simulation with bigger amount of atoms

import time 
import argparse

import torch

from nnmd.nn import Neural_Network
from nnmd.nn.dataset import make_atomic_dataset
from nnmd.util import gpaw_parser

params_parser = argparse.ArgumentParser(description = "Test code for pre-trained neural network\
                                                       on gpaw simulation with bigger amount of atoms")
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

# Atomic NN, nodes amount in hidden layers
hidden_nodes = [16, 8]
# create an instance
net = Neural_Network(hidden_nodes = hidden_nodes,
                     use_cuda = use_cuda)
print("done")

device = torch.device('cuda') if use_cuda else torch.device('cpu')
dtype = torch.float32

print(f"Move NN to {'GPU' if device.type == 'cuda' else 'CPU'}:", end = ' ')
# move NN to right device 
net.to(device = device)
print("done")

print(f"Make dataset:", end = ' ')

# params of symmetric functions
rc = 12.0
eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3

cartesians = torch.as_tensor(cartesians, device = device, dtype = dtype)
e_dft = torch.as_tensor(e_dft, device = device, dtype = dtype)
f_dft = torch.as_tensor(f_dft, device = device, dtype = dtype)
dataset = make_atomic_dataset(cartesians, rc, eta, rs, k, _lambda, xi, device, e_dft, f_dft)
print("done")

load_models = True
path = 'models'

print("Config subnets:", end = ' ')

net.config(n_atoms, load_models = load_models, path = path)
time.sleep(1)
print("done")

print("Testing:", end = ' ')

start = time.time()
e_nn, f_nn = net.predict(dataset)
end = time.time()

print("done")
print(f"\nTesting sample size: {len(dataset)} ")
print(f"Testing time ({'GPU' if device.type == 'cuda' else 'CPU'}): {(end - start):.3f} s")

test_loss, test_e_loss, test_f_loss = net.loss(e_nn, e_dft, f_nn, f_dft)

test_e_loss = test_e_loss.cpu().detach().numpy()
test_f_loss = test_f_loss.cpu().detach().numpy()
test_loss = test_loss.cpu().detach().numpy()

test_loss_info = f"test: RMSE E = {test_e_loss:.4f}, RMSE F = {test_f_loss:.4f}, RMSE total = {test_loss:.4f} eV"
print(test_loss_info)