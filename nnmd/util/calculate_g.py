import torch
import nnmd_cpp

def calculate_g(cartesians: torch.Tensor, device: torch.device,
                symm_func_params: dict[str, float]):
        """Calculates symmetric functions for each structs of atoms with specified parameters.

        Args:
            cartesians (torch.Tensor): structs of atoms (atomic systems in certain time moment)
            device (torch.device): target device
            symm_func_params (dict[str, float]): parameters of symmetric functions
            
        """
        nnmd = nnmd_cpp.cuda if device.type == 'cuda' else nnmd_cpp.cpu
        
        g = []

        # calculate symmetric functions values for each struct of atoms
        for struct in cartesians:
            g_struct = nnmd.calculate_sf(struct, symm_func_params['r_cutoff'],
                                                 symm_func_params['eta'],
                                                 symm_func_params['rs'],
                                                 symm_func_params['k'],
                                                 symm_func_params['lambda'],
                                                 symm_func_params['xi'])
            g.append(g_struct)
        
        return torch.stack(g).to(device = device, dtype = torch.float32)