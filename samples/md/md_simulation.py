#!/usr/bin/env python
# -*- coding: utf-8 -*-

# example of nnmd package usage with gpaw simulation

import os
import shutil
import time 
import argparse

import torch

import numpy as np
import pandas as pd

from nnmd.nn import Neural_Network
from nnmd.util import gpaw_parser
from nnmd.md import MDSimulation

import ase.data as ad

params_parser = argparse.ArgumentParser(description = "Script for MD simulation with neural network")   
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

# params of symmetric functions
symm_func_params = {"r_cutoff": 12.0,
                    "eta": 0.01,
                    "k": 1,
                    "rs": 0.5,
                    "lambda": 1,
                    "xi": 3}
h = 0.1

# mass of 
m_atom = ad.atomic_masses[ad.atomic_numbers['Cu']]
# temperature of system (in Kelvin)
T = 300.0
# Van der Waals radius
rVan = 3.8
dt = 10e-15
h = 0.01

# inputs and targets
sep = int(0.8 * n_structs)
data = np.array(cartesians[sep + 1:])
np.savez("cartesians_actual.npy.npz", x = data[:, :, 0], y = data[:, :, 1], z = data[:, :, 2])

# start data: initial positions, forces (=> acceleration) and velocity
cartesians_initial = torch.as_tensor(cartesians[sep], dtype = dtype)
f_dft = torch.as_tensor(f_dft[sep], dtype = dtype)
v_initial = (f_dft / m_atom) * dt

nn = Neural_Network()
hidden_nodes = [16, 8]
nn.config(hidden_nodes = hidden_nodes,
          use_cuda = use_cuda,
          n_atoms = n_atoms,
          load_models = True, path = "../gpaw_simulation/models")

md_system = MDSimulation(N_atoms = n_atoms, cartesians = cartesians_initial, nn = nn,
                         mass = m_atom, rVan = rVan, symm_func_params = symm_func_params,
                         L = h, T = T, dt = dt, h = h, v_initial = v_initial)

# MD simulation
md_system.run_md_simulation(steps = 50)

data = np.array(md_system.cartesians_history)
np.savez("cartesians_history.npy.npz", x = data[:, :, 0], y = data[:, :, 1], z = data[:, :, 2])