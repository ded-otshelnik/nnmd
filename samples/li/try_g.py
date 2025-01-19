# This script is used to test the g function in the nnmd package

from ase.io import Trajectory
import torch
torch.manual_seed(0)
import numpy as np
from nnmd.features import calculate_sf

if __name__ == "__main__":
    traj = Trajectory('input/Li_crystal_27.traj')
    positions = np.array([atoms.positions for atoms in traj])

    r_cutoff = 3.54
    n_radial = 4
    n_angular = 4
    features = [2] * n_radial + [4] * n_angular
    params_g2 = [
            [r_cutoff, 2, 0.1],
            [r_cutoff, 2, 0.3],
            [r_cutoff, 2, 0.5],
            [r_cutoff, 2, 0.7]
    ]
    params_g4 = [
            [r_cutoff, 1.0, 3, -1],
            [r_cutoff, 1.0, 3, 1],
            [r_cutoff, 2.0, 3, -1],
            [r_cutoff, 2.0, 3, 1]
    ]

    params = params_g2 + params_g4
    symm_funcs_data = {'features': features, 'params': params}

    cell = torch.tensor(traj[0].cell.array, dtype = torch.float32)
    cartesians = torch.tensor(positions, dtype = torch.float32)
    cartesians.requires_grad = True

    g, dg = calculate_sf(cartesians, cell, symm_funcs_data)
    print(g)
    print(dg)