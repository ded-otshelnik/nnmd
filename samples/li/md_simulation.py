#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MD simulation of Cu atoms with neural network
import argparse

import torch

import numpy as np

from nnmd.nn import HDNN
from nnmd.md import MDSimulation

import ase.data as ad

params_parser = argparse.ArgumentParser(description = "Script for MD simulation with neural network")   
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

# params of symmetric functions
symm_func_params = {"r_cutoff": 6.0,
                    "eta": 0.01,
                    "k": 1,
                    "rs": 0.5,
                    "lambda": -1,
                    "xi": 3}
h = 10e-1

# mass of atom (in atomic mass units)
m_atom = ad.atomic_masses[ad.atomic_numbers['Li']]
# temperature of system (in Kelvin)
T = 300.0
# size of the box
L = 9.0
# Van der Waals radius
rVan = 1.44
dt = 10e-3

# start data: initial positions, forces (=> acceleration) and velocity
cartesians = np.load("cartesians_actual.npy.npz")['cartesians']
forces = np.load("forces_actual.npy.npz")['forces']

start = 0
v_initial = np.load("velocities_actual.npy.npz")['velocities'][start]
cartesians_initial = cartesians[start]
n_atoms = cartesians_initial.shape[0]
a_initial = forces[start] / m_atom

nn = HDNN()
hidden_nodes = [30, 30]
nn.config(hidden_nodes = hidden_nodes,
          use_cuda = use_cuda,
          n_atoms = n_atoms,
          load_models = True, path = "../li/models/atomic_nn_Li.pt")

md_system = MDSimulation(N_atoms = n_atoms, cartesians = cartesians_initial, nn = nn,
                         mass = m_atom, rVan = rVan, symm_func_params = symm_func_params,
                         L = L, T = T, dt = dt, h = h, v_initial = v_initial, a_initial = a_initial)

# MD simulation
md_system.run_md_simulation(steps = 600)

data = np.array(md_system.cartesians_history)
np.savez("cartesians_history.npy.npz", x = data[:, :, 0], y = data[:, :, 1], z = data[:, :, 2])
np.savez("forces_history.npy.npz", forces = np.array(md_system.forces_history))
np.savez("velocities_history.npy.npz", velocities = np.array(md_system.velocities_history))