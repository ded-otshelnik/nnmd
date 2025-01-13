import torch
from .symm_funcs import calculate_distances, calculate_cosines, g2_function, g4_function, g5_function, calculate_mask, SymmetryFunction

from tqdm import tqdm

from joblib import Parallel, delayed, cpu_count

def calculate_sf(cartesians: torch.Tensor, symm_funcs_data: dict) -> torch.Tensor:
    """
    Calculate symmetry functions for a batch of molecules.
    Args:
        cartesians: torch.Tensor of shape (n_molecules, n_atoms, 3)
        symm_funcs_data: dict - dictionary with symmetry functions data
    Returns:
        torch.Tensor of shape (n_molecules, n_atoms, n_symm_funcs)
    """
    def _internal(cart, symm_funcs_data):
        # Calculate all necessary distances, cosines and mask
        distances = calculate_distances(cart)
        rij = distances.unsqueeze(2)
        rik = distances.unsqueeze(1)
        rjk = distances.unsqueeze(0)
        cosines = calculate_cosines(rij, rik, rjk)
        mask = calculate_mask(cart)
        
        g_struct = []
        dg_struct = []
        for g_func, g_params in zip(symm_funcs_data['features'], symm_funcs_data['params']):

            if g_func == SymmetryFunction.G2.value:
                g_values = g2_function(distances, g_params[0], g_params[1], g_params[2])
            
            elif g_func == SymmetryFunction.G4.value:
                g_values = g4_function(rij, rik, rjk,
                                    cosines, g_params[0], g_params[1], g_params[2], g_params[3], mask)
                
            elif g_func == SymmetryFunction.G5.value:
                g_values = g5_function(rij, rik,
                                    cosines, g_params[0], g_params[1], g_params[2], g_params[3], mask)
            else:
                raise ValueError(f"Unknown symmetry function number: {g_func}")
            
            # Calculate the derivatives of the symmetry functions
            dg_values = torch.autograd.grad(g_values.sum(), cart, retain_graph = True)[0]

            # Normalize the symmetry functions
            g_values = (g_values - g_values.min()) / (g_values.max() - g_values.min())

            dg_struct.append(dg_values)
            g_struct.append(g_values)

        # Stack the symmetry functions and their derivatives
        dg_struct = torch.stack(dg_struct, dim = -1)
        g_struct = torch.stack(g_struct, dim = -1)

        # Swap the last two dimensions of the derivative tensor
        # to match the shape of the symmetry functions tensor
        dg_struct = dg_struct.permute(0, 2, 1)
        return g_struct, dg_struct
    r = Parallel(n_jobs = cpu_count())(delayed(_internal)(cart, symm_funcs_data) for cart in tqdm(cartesians))
    #r = [_internal(cart, symm_funcs_data) for cart in tqdm(cartesians)]  
    g, dG = zip(*r)
    g = torch.stack(g)
    dG = torch.stack(dG)
    return g, dG