import torch
import nnmd_cpp

def calculate_g(cartesians: torch.Tensor, r_cutoff: float,
                eta: float, rs: float, k: float, _lambda: int, xi: float,
                device: torch.device):
        """Calculates symmetric functions for each structs of atoms with specified parameters.

        Args:
            cartesians (torch.Tensor): structs of atoms (atomic systems in certain time moment)
            r_cutoff (float): cutoff radius
            eta (float): parameter of symmetric functions
            rs (float): parameter of symmetric functions
            k (float): parameter of symmetric functions
            _lambda (int): parameter of symmetric functions
            xi (float): parameter of symmetric functions
            
        """
        nnmd = nnmd_cpp.cuda if device.type == 'cuda' else nnmd_cpp.cpu
        # arrays of g and their derivatives
        g = []
        # params of symmetric functions
        # calculate symmetric functions values for each struct of atoms and its derivatives
        for struct in cartesians:
            g_struct = nnmd.calculate_sf(struct, r_cutoff, eta, rs, k, _lambda, xi)
            g.append(g_struct)
        
        # g values - inputs of Atomic NNs
        # so we need to store gradient for backpropagation
        g = torch.stack(g).to(device = device, dtype = torch.float32)
        return g