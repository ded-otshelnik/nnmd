#!/usr/bin/env python
# -*- coding: utf-8 -*-

# example of nnmd package usage with lennard-jones potentials


import time 
import argparse

import torch

from nnmd.nn import Neural_Network

from lennard_jones import lennard_jones_gen

params_parser = argparse.ArgumentParser(description = "Sample code of nnmd usage")
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

print("Getting Lennard-Jones potential: ", end = '')

cartesians, e_dft, f_dft, distances = lennard_jones_gen()

print("done")

print("Create an instance of NN:", end = ' ')
# Global parameters of HDNN 
epochs = 100
batch_size = len(distances)
# Atomic NN, nodes amount in hidden layers
hidden_nodes = [10, 10]
# create an instance
net = Neural_Network(input_nodes = 1, 
                     hidden_nodes = hidden_nodes,
                     epochs = epochs,
                     use_cuda = use_cuda)
print("done")

device = torch.device('cuda') if use_cuda else torch.device('cpu')
print(f"Move NN to {'GPU' if device.type == 'cuda' else 'CPU'}:", end = ' ')
# move NN to right device 
net.to(device = device)
print("done")

print("Config subnets and prepare dataset:", end = ' ')
# prepare data for nets and its subnets
rc = 2.5
eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3
n_structs = len(cartesians)
n_atoms = len(cartesians[0])

# sep = int(0.8 * n_structs)
# batch_size = sep 

net.compile(cartesians, len(cartesians), n_atoms, rc, eta, rs, k, _lambda, xi)

print("done")

print("Training:")
start = time.time()

net.fit(e_dft, batch_size)

end = time.time()
train_time = (end - start)
net.net_log.info(f"Training time ({'GPU' if device.type == 'cuda' else 'CPU'}): {train_time:.3f} s")

import matplotlib.pyplot as plt

e_train = net.predict(cartesians).cpu()
print(e_dft)
print(e_train)

plt.plot(distances, e_dft)
plt.plot(distances, e_train.sum(1))
plt.show()