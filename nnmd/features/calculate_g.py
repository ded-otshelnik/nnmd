import torch
import nnmd_cpp

from tqdm import tqdm

def calculate_g(cartesians: torch.Tensor, 
                symm_func_params: dict):
        """Calculates symmetric functions for each structs of atoms with specified parameters.

        Args:
            cartesians (torch.Tensor): structs of atoms (atomic systems in certain time moment)
            device (torch.device): target device
            symm_func_params (dict[str, float]): parameters of symmetric functions
            
        """
        nnmd = nnmd_cpp.cuda if cartesians.device.type == 'cuda' else nnmd_cpp.cpu
        
        g = []

        # calculate symmetric functions and its derivatives for each struct of atoms
        for struct in tqdm(cartesians):
            g_struct = nnmd.calculate_sf(struct, symm_func_params['features'], symm_func_params['params'])
            torch.cuda.empty_cache()
            g.append(g_struct)
        g = torch.stack(g).to(device = cartesians.device, dtype = torch.float32)

        return g