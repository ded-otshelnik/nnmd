import torch
from torch.utils.data import Dataset

from ..util import calculate_g

class AtomicDataset(Dataset):
    def __init__(self, cartesians: torch.Tensor,
                 symm_func_params: dict[str, float],
                 g: torch.Tensor,
                 h: float) -> None:

        self.cartesians = cartesians
        self.symm_func_params = symm_func_params
        self.g = g
        self.h = h
        self.len = self.cartesians.size(0)
    
    def __getitem__(self, index):
        return self.cartesians[index], self.g[index]
    
    def __len__(self):
        return self.len
    
class TrainAtomicDataset(AtomicDataset):
    def __init__(self, cartesians: torch.Tensor, symm_func_params: dict[str, float],
                 g: torch.Tensor, h: float, e_dft: torch.Tensor, f_dft: torch.Tensor) -> None:
                 
        super().__init__(cartesians,symm_func_params, g, h)
        self.g.requires_grad = True
        self.e_dft = e_dft
        self.f_dft = f_dft
    
    def __getitem__(self, index):
        return self.cartesians[index], self.g[index], self.e_dft[index], self.f_dft[index]
    
def make_atomic_dataset(cartesians: torch.Tensor, symm_func_params: dict[str, float], h: float, device: torch.device,
                        e_dft: torch.Tensor = None, f_dft: torch.Tensor = None, train: bool = False) -> AtomicDataset:
    """Function creates atomic dataset with computation of symmetric functions
    according input data.

    Args:
        cartesians (torch.Tensor): positions of atoms
        symm_func_params (dict[str, float]): parameters of symmetric functions
        h (float): step of coordinate-wise moving (used in forces caclulations). Defaults to 1.
        device (torch.device): target device
        e_dft (torch.Tensor): target energies
        f_dft (torch.Tensor): target atomic forces
        train (bool): enables gradient computation if it is a train dataset. Defaults to False

    Raises:
        ValueError: if a cartesians storage is not a tensor
    """

    # cartesians must be presented as a tensor
    # because operation of calculating is tensor-like
    if not isinstance(cartesians, torch.Tensor):
        raise ValueError(f"Cartesians must be tensors, not {type(cartesians)}")
    
    g = calculate_g(cartesians, device, symm_func_params)
    if train:
        return TrainAtomicDataset(cartesians, symm_func_params, g ,h, e_dft, f_dft)
    else:
        return AtomicDataset(cartesians, symm_func_params, g, h)