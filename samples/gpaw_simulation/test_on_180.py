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
 
use_cuda = (args.use_cuda is not None) and torch.cuda.is_available()
if use_cuda:
    print("GPU usage is enabled")
elif args.use_cuda and not torch.cuda.is_available():
    print("Pytorch compiled/downloaded without CUDA support. GPU is disabled")
else:
    print("GPU usage is disabled")
device = torch.device('cuda') if use_cuda else torch.device('cpu')
dtype = torch.float32

print("Get info from GPAW simulation: ", end = '')
n_structs, n_atoms, cartesians, f_dft, e_dft = gpaw_parser(args.gpaw_file)
print("done", f"Make dataset:", end = ' ', sep = '\n')

# params of symmetric functions
symm_func_params = {"r_cutoff": 6.0,
                    "eta": 0.01,
                    "k": 1,
                    "rs": 0.5,
                    "lambda": -1,
                    "xi": 3}
h = 0.01

cartesians = torch.as_tensor(cartesians, device = device, dtype = dtype)
e_dft = torch.as_tensor(e_dft, device = device, dtype = dtype)
f_dft = torch.as_tensor(f_dft, device = device, dtype = dtype)
dataset = make_atomic_dataset(cartesians, symm_func_params, h, device)
print("done")

load_models = True
path = 'models_new'

input_nodes = 5
hidden_nodes = [30, 30]

print("Create an instance of NN and config its subnets:", end = ' ')
net = Neural_Network()
net.config(hidden_nodes = hidden_nodes, 
           use_cuda = use_cuda,
           dtype = dtype,
           n_atoms = n_atoms,
           input_nodes = input_nodes,
           load_models = load_models,
           path = path)
print("done")

print("Testing:", end = ' ')

start = time.time()
test_loss, test_e_loss, test_f_loss = 0, 0, 0
for cartesian, e_dft, f_dft in zip(cartesians, e_dft, f_dft):
    test_e_nn, test_f_nn = net.predict(cartesian, symm_func_params, h)
    loss = net.loss(test_e_nn, e_dft.unsqueeze(0), test_f_nn, f_dft)
    test_loss += loss[0]
    test_e_loss += loss[1]
    test_f_loss+= loss[2]
end = time.time()

print("done", f"Testing sample size: {len(dataset)} ",
        f"Testing time ({'GPU' if device.type == 'cuda' else 'CPU'}): {(end - start):.3f} s",
        sep = '\n')

test_e_loss = test_e_loss.cpu().detach().numpy() / n_structs
test_f_loss = test_f_loss.cpu().detach().numpy() / n_structs
test_loss = test_loss.cpu().detach().numpy() / n_structs

print(f"test: RMSE E = {test_e_loss:.4f}, RMSE F = {test_f_loss:.4f}, RMSE total = {test_loss:.4f} eV")