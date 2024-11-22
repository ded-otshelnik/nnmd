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
h = 0.01

# mass of atom of Copper (in atomic mass units)
m_atom = ad.atomic_masses[ad.atomic_numbers['Cu']]
# temperature of system (in Kelvin)
T = 300.0
# Van der Waals radius
rVan = 1.84
dt = 10e-15

# start data: initial positions, forces (=> acceleration) and velocity
cartesians = torch.as_tensor(np.load("cartesians_actual.npy.npz")['cartesians'], dtype = dtype)
forces = torch.as_tensor(np.load("forces_actual.npy.npz")['forces'], dtype = dtype)

start = int(0.8 * len(cartesians))
v_initial = torch.as_tensor(np.load("velocities_actual.npy.npz")['velocities'][start], dtype = dtype)
cartesians_initial = cartesians[start]
n_atoms = cartesians_initial.shape[0]
a_initial = forces[start] / m_atom

# Vm = np.sqrt(8 * 8.314462 * T / (np.pi * m_atom))
# v_initial = torch.as_tensor(np.log(np.random.uniform(0.5, 2.5, (n_atoms, 3))) * Vm, dtype = dtype)
# a_initial = torch.zeros((n_atoms, 3), dtype = dtype)

nn = HDNN()
hidden_nodes = [30, 30]
nn.config(hidden_nodes = hidden_nodes,
          use_cuda = use_cuda,
          n_atoms = n_atoms,
          load_models = True, path = "../li/checkpoint.pt")

md_system = MDSimulation(N_atoms = n_atoms, cartesians = cartesians_initial, nn = nn,
                         mass = m_atom, rVan = rVan, symm_func_params = symm_func_params,
                         L = h, T = T, dt = dt, h = h, v_initial = v_initial, a_initial = a_initial)

# MD simulation
md_system.run_md_simulation(steps = 10)

data = np.array(md_system.cartesians_history)
np.savez("cartesians_history.npy.npz", x = data[:, :, 0], y = data[:, :, 1], z = data[:, :, 2])
np.savez("forces_history.npy.npz", forces = np.array(md_system.forces_history))
np.savez("velocities_history.npy.npz", velocities = np.array(md_system.velocities_history))