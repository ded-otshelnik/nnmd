import torch
import nnmd_cpp

def calculate_dg(cartesians: torch.Tensor, g: torch.Tensor, symm_funcs_data: dict):
    """Calculates derivatives of symmetric functions for each structs of atoms with specified parameters.

    Args:
        cartesians (torch.Tensor): structs of atoms (atomic systems in certain time moment)
        g (torch.Tensor): symmetric functions for each structs of atoms
        symm_func_params (dict): parameters of symmetric functions
        
    """
    nnmd = nnmd_cpp.cuda if cartesians.device.type == 'cuda' else nnmd_cpp.cpu
    dg = nnmd.calculate_dG(cartesians, g, symm_funcs_data['features'], symm_funcs_data['params'], 1.0)
    return dg
    