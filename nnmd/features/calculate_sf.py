import functools
import math
import torch
from functorch import vmap, grad
from .symm_funcs import calculate_distances, g2_function, g4_function, g5_function, SymmetryFunction

from joblib import Parallel, cpu_count, delayed
from torch.autograd import detect_anomaly

from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _internal(cart: torch.Tensor, cell: torch.Tensor, symm_funcs_data: dict) -> torch.Tensor:
    """
    Calculate symmetry functions for a single molecule.
    Args:
        cart: torch.Tensor of shape (n_atoms, 3)
        cell: torch.Tensor of shape (3, 3)
        symm_funcs_data: dict - dictionary with symmetry functions data
    Returns:
        torch.Tensor of shape (n_atoms, n_symm_funcs)
    """
    # Calculate distances and triplets
    distances, triplets = calculate_distances(cart, cell)
        
    g_struct = []
    dg_struct = []
    for g_func, g_params in zip(symm_funcs_data['features'], symm_funcs_data['params']):
        if g_func == SymmetryFunction.G2.value:
            # g_params = [cutoff, eta, rs]
            g_values = g2_function(distances, g_params[0], g_params[1], g_params[2])
            
        elif g_func == SymmetryFunction.G4.value:
            # g_params = [cutoff, eta, zeta, lambd]
            g_values = g4_function(distances, triplets, g_params[0], g_params[1], g_params[2], g_params[3])
                
        elif g_func == SymmetryFunction.G5.value:
            # g_params = [cutoff, eta, zeta, lambd]
            g_values = g5_function(distances, triplets, g_params[0], g_params[1], g_params[2], g_params[3])
        else:
            raise ValueError(f"Unknown symmetry function number: {g_func}")

        # Calculate the gradient of the symmetry function
        dg_values = torch.autograd.grad(g_values.sum(), cart, create_graph = True)[0]

        g_struct.append(g_values.detach())
        dg_struct.append(dg_values.detach())

    # Stack the symmetry functions and their derivatives
    g_struct = torch.stack(g_struct, dim = -1)
    dg_struct = torch.stack(dg_struct, dim = -1)

    # Permute the symmetry functions to the shape (n_atoms, n_symm_funcs, 3)
    dg_struct = dg_struct.permute(0, 2, 1)
    return g_struct, dg_struct

def calculate_sf(cartesians: torch.Tensor, cell: torch.Tensor, symm_funcs_data: dict) -> torch.Tensor:
    """
    Calculate symmetry functions for a batch of molecules.
    Args:
        cartesians: torch.Tensor of shape (n_molecules, n_atoms, 3)
        symm_funcs_data: dict - dictionary with symmetry functions data
    Returns:
        torch.Tensor of shape (n_molecules, n_atoms, n_symm_funcs)
    """
    op = functools.partial(_internal, cell = cell, symm_funcs_data = symm_funcs_data)
    
    set_seed(42)
    cartesians.requires_grad_(True)
    r = [op(cart) for cart in tqdm(cartesians)]
    g, dg = zip(*r)
            
    g = torch.stack(g, dim = 0)
    dg = torch.stack(dg, dim = 0)
    return g, dg