import torch
from torch.utils.data import Dataset

from ..util import calculate_g

class TrainAtomicDataset(Dataset):
    def __init__(self, cartesians: torch.Tensor, r_cutoff: float,
                 eta: float, rs: float, k: float, _lambda: float, xi: float,
                 g: torch.Tensor, e_dft: torch.Tensor, f_dft: torch.Tensor) -> None:
                 
        self.cartesians = cartesians
        self.r_cutoff, self.eta, self.rs, self.k, self._lambda, self.xi = r_cutoff, eta, rs, k, _lambda, xi
        self.g = g
        self.e_dft = e_dft
        self.f_dft = f_dft
        self.len = self.cartesians.size(0)
    
    def __getitem__(self, index):
        return self.cartesians[index], self.g[index], self.e_dft[index], self.f_dft[index]
    
    def __len__(self):
        return self.len
class AtomicDataset(Dataset):
    def __init__(self, cartesians: torch.Tensor, r_cutoff: float,
                 eta: float, rs: float, k: float, _lambda: float, xi: float,
                 g: torch.Tensor) -> None:
                 
        self.cartesians = cartesians
        self.r_cutoff, self.eta, self.rs, self.k, self._lambda, self.xi = r_cutoff, eta, rs, k, _lambda, xi
        self.g = g
        self.len = self.cartesians.size(0)
    
    def __getitem__(self, index):
        return self.cartesians[index], self.g[index]
    
    def __len__(self):
        return self.len
    
def make_atomic_dataset(cartesians: torch.Tensor, r_cutoff: float,
                        eta: float, rs: float, k: float, _lambda: float, xi: float, device: torch.device,
                        e_dft: torch.Tensor = None, f_dft: torch.Tensor = None, train: bool = True):
    """Function creates atomic dataset with calculation of symmetric functions
    according input data. It can make dataset for training
    and for 'pure' (without energies and forces) calculations (parameter 'train').

    Args:
        cartesians (torch.Tensor): positions of atoms
        r_cutoff (float): cutoff radius
        eta (float): parameter of symmetric functions
        rs (float): parameter of symmetric functions
        k (float): parameter of symmetric functions
        _lambda (float): parameter of symmetric functions
        xi (float): parameter of symmetric functions
        device (torch.device): parameter of symmetric functions
        e_dft (torch.Tensor): parameter of symmetric functions. Defaults to None
        f_dft (torch.Tensor): parameter of symmetric functions. Defaults to None

    Raises:
        ValueError: if a cartesians storage is not a tensor

    Returns:
        _type_: _description_
    """
    # cartesians must be presented as a tensor
    # because operation of calculating is tensor-like
    if not isinstance(cartesians, torch.Tensor):
        raise ValueError(f"Cartesians must be tensors, not {type(cartesians)}")
    
    g = calculate_g(cartesians, r_cutoff, eta, rs, k, _lambda, xi, device)
    if train:
        dataset = TrainAtomicDataset(cartesians, r_cutoff, eta, rs, k, _lambda, xi, g, e_dft, f_dft)
    else:
        dataset = AtomicDataset(cartesians, r_cutoff, eta, rs, k, _lambda, xi, g)

    return dataset

