from nnmd.features.symm_funcs import calculate_distances
import torch
import numpy as np

from ase.io import Trajectory

from nnmd.features import params_for_G2, params_for_G4, calculate_sf

if __name__ == "__main__":
    traj = Trajectory('input/Li_crystal_27.traj')
    cartesians = torch.tensor(np.array([atoms.get_positions() for atoms in traj]), dtype = torch.float32)
    cartesians.requires_grad = True
    cartesians = cartesians.to(device = 'cuda')
    
    distances = calculate_distances(cartesians[0])
    
    max_dist = torch.max(distances)
    print("Max distance: ", max_dist)
    r_cutoff_g2 = max_dist.item() / 2 
    r_cutoff_g4 = max_dist.item() / 2
    n_radial = 6
    n_angular = 6

    features = [2] * n_radial + [4] * n_angular
    params = params_for_G2(n_radial, r_cutoff_g2) + params_for_G4(n_angular, r_cutoff_g4)
    symm_funcs_data = {'features': features, 'params': params}

    g, dg = calculate_sf(cartesians[:1], symm_funcs_data)
    print("G and dG by python code:")
    print(g.cpu().detach().numpy())
    print(dg.cpu().detach().numpy())