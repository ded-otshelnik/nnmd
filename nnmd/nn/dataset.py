from math import exp
import torch
from torch.utils.data import Dataset, Subset
import numpy as np

from ..util import calculate_g

# C++/CUDA extention
import nnmd_cpp
class AtomicDataset(Dataset):
    """Class represents atomic dataset with symmetric functions
    and atomic positions.

    Args:

        dataset (Dataset): dataset with atomic positions and symmetric functions
        symm_func_params (dict[str, float]): parameters of symmetric functions
        h (float): step of coordinate-wise moving (used in forces caclulations).
    """
    def __init__(self, cartesians: torch.Tensor, 
                energies: torch.Tensor, forces: torch.Tensor,
                symm_func_params: dict[str, float]) -> None:
        self.cartesians: torch.Tensor = cartesians
        self.energies: torch.Tensor = energies
        self.forces: torch.Tensor = forces

        self.symm_func_params: dict[str, float] = symm_func_params
        self.len: int = len(self.cartesians)
    
    def __getitem__(self, index):
        return self.cartesians[index], self.energies[index], self.forces[index]
    
    def __len__(self):
        return self.len
    
class TrainAtomicDataset(AtomicDataset):
    def __init__(self, cartesians: torch.Tensor, g: torch.Tensor, dG: torch.Tensor,
                energies: torch.Tensor, forces: torch.Tensor,
                symm_func_params: dict[str, float],) -> None:
                 
        super().__init__(cartesians, energies, forces, symm_func_params)
        self.g: torch.Tensor = g
        self.dG: torch.Tensor = dG
        # enable gradient computation
        # because they are inputs of NN
        self.g.requires_grad = True
        assert len(self.cartesians) == len(self.g) == len(self.dG) == len(self.energies) == len(self.forces)
        assert self.cartesians.device == self.g.device == self.dG.device == self.energies.device == self.forces.device

    def __getitem__(self, index):
        return self.g[index], self.dG[index], self.energies[index], self.forces[index]
    
def make_atomic_dataset(dataset: Subset,
                        symm_func_params: dict[str, float],
                        device: torch.device, train: bool = False, **kwargs) -> AtomicDataset:
    """Create atomic dataset with symmetric functions.

    Args:

        dataset (Dataset): dataset with atomic positions, energies and forces
        symm_func_params (dict[str, float]): parameters of symmetric functions
        device (torch.device): device to store data
        train (bool): if True, create dataset for training
        path (str): path to file with precalculated dG

    Returns:

        AtomicDataset: atomic dataset with symmetric functions
    """
    # retrieve cartesians, energies, forces from subset and get symmetric functions values
    dataset_indices = dataset.indices
    cartesians = dataset.dataset.tensors[0][dataset_indices]
    energies = dataset.dataset.tensors[1][dataset_indices]
    forces = dataset.dataset.tensors[2][dataset_indices]
    # if dataset will be used for training calculate g
    if train:
        if 'g' in kwargs and 'dG' in kwargs:
            g = torch.load(kwargs['g'])
            dG = torch.load(kwargs['dG'])
        else:
            g, dG = calculate_g(cartesians, device, symm_func_params)
            torch.save(g, f"g_{kwargs['path']}.pt")
            torch.save(dG, f"dG_{kwargs['path']}.pt")
        
        return TrainAtomicDataset(cartesians, g, dG, energies, forces, symm_func_params)
    # otherwise return just AtomicDataset
    else:
        return AtomicDataset(cartesians, energies, forces, symm_func_params)