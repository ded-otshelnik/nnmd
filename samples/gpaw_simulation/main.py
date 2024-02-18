#!/usr/bin/env python
# -*- coding: utf-8 -*-

# example of nnmd package usage with gpaw simulation

import os
import shutil
import time 
import argparse

import torch

from nnmd.nn import Neural_Network
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
# elif use_cuda and not torch.cuda.is_available():
    print("Pytorch compiled/downloaded without CUDA support. GPU is disabled")
else:
    print("GPU usage is disabled")

print("Getting info from GPAW simulation: ", end = '')
n_structs, n_atoms, cartesians, f_dft, e_dft = gpaw_parser(args.gpaw_file)
print("done")

print("Create an instance of NN:", end = ' ')
# Global parameters of HDNN 
epochs = 10
batch_size = 16
# Atomic NN, nodes amount in hidden layers
hidden_nodes = [16, 8]
# create an instance
net = Neural_Network(hidden_nodes = hidden_nodes,
                     epochs = epochs,
                     use_cuda = use_cuda)
print("done")

device = torch.device('cuda') if use_cuda else torch.device('cpu')
print(f"Move NN to {'GPU' if device.type == 'cuda' else 'CPU'}:", end = ' ')
# move NN to right device 
net.to(device=device)
print("done")

print(f"Separate data to train and test datasets:", end = ' ')
# ~80% - train, ~20% - test
sep = int(0.8 * n_structs)
train_data, test_data = cartesians[:sep], cartesians[sep:]
train_e_dft, test_e_dft = e_dft[:sep], e_dft[sep:]
train_f_dft, test_f_dft = f_dft[:sep], f_dft[sep:]
print("done")
# params that define what NN will do 
# train model
train = True
# test model
test = True
# save model params as files in <path> directory
save = True
path = 'models'

print("Config subnets and prepare dataset:", end = ' ')
# prepare data for nets and its subnets
rc = 12.0
eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3
net.compile(train_data, len(train_data), len(train_data[0]), rc, eta, rs, k, _lambda, xi)
print("done")
if train:
    print("Training:")
    start = time.time()

    net.fit(train_e_dft, train_f_dft, batch_size)

    end = time.time()
    train_time = (end - start)
    net.net_log.info(f"Training time ({'GPU' if device.type == 'cuda' else 'CPU'}): {train_time:.3f} s")

if test:
    print("Testing:")
    test_e_dft, test_f_dft = torch.tensor(test_e_dft, device = device), torch.tensor(test_f_dft, device = device)
    start = time.time()

    test_e_nn, test_f_nn = net.predict(test_data)

    end = time.time()
    net.net_log.info(f"\nTesting time ({'GPU' if device.type == 'cuda' else 'CPU'}): {(end - start):.3f} s")
    
    test_loss, test_e_loss, test_f_loss = net.loss(test_e_nn, test_e_dft, test_f_nn, test_f_dft)

    test_e_loss = test_e_loss.cpu().detach().numpy()
    test_f_loss = test_f_loss.cpu().detach().numpy()
    test_loss = test_loss.cpu().detach().numpy()

    test_loss_info = f"test: RMSE E = {test_e_loss:.4f}, RMSE F = {test_f_loss:.4f}, RMSE total = {test_loss:.4f} eV"
    net.net_log.info(test_loss_info)

if save:
    print("Saving model: ", end = '')

    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors = True)
    os.mkdir(path)

    net.save_model(path)
    print("done")