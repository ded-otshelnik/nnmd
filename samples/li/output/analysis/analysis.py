import argparse

import numpy as np

import torch

from nnmd.util import traj_parser
from nnmd.util import calculate_g

from nnmd import nnmd_cpp

# import warnings
# warnings.filterwarnings('ignore')

params_parser = argparse.ArgumentParser(description = "Sample code of nnmd usage")
params_parser.add_argument("data_file", type = str, help = "Path to file with atomic data")
args = params_parser.parse_args()

device = torch.device('cuda')
dtype = torch.float32
cartesians, forces, energies, velocities = traj_parser(args.data_file)


cartesians = torch.as_tensor(cartesians, dtype = dtype, device = device)

# euclidean distances between atoms in the first structure
distances = torch.norm(cartesians[0, :, None, :] - cartesians[0, None, :, :], dim = -1, p = 2)
# all distances that are 0 will be replaced with inf so that they do not affect the minimum
min_distance = torch.where(distances == 0, torch.as_tensor(np.inf, device = device), distances).min().item()
print("Min distance: ", min_distance)

# params of symmetric functions
symm_func_params = {"r_cutoff": 2 * min_distance,
                    "eta": 0.333,
                    "k": 2,
                    "rs": 3,
                    "lambda": 1,
                    "xi": 2}
h = 2

n_structs = len(cartesians)
n_atoms = len(cartesians[0])

g = calculate_g(cartesians, device, symm_func_params)

nnmd = nnmd_cpp.cuda if device.type == 'cuda' else nnmd_cpp.cpu
dg = nnmd.calculate_dG(cartesians, g,
                        symm_func_params['r_cutoff'],
                        symm_func_params['eta'],
                        symm_func_params['rs'],
                        symm_func_params['k'],
                        symm_func_params['lambda'],
                        symm_func_params['xi'], h)
np.save("g.npy", g.cpu().detach().numpy())
np.save("dg.npy", dg.cpu().detach().numpy())