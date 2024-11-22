import torch
from torch.utils.data import Dataset, Subset

from ..util import calculate_g

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
                symm_func_params: dict[str, float],
                h: float) -> None:
        self.cartesians = cartesians
        self.energies = energies
        self.forces = forces

        self.symm_func_params = symm_func_params
        self.h = h
        self.len = len(self.cartesians)
    
    def __getitem__(self, index):
        return self.cartesians[index], self.energies[index], self.forces[index]
    
    def __len__(self):
        return self.len
    
class TrainAtomicDataset(AtomicDataset):
    def __init__(self, cartesians: torch.Tensor, g: torch.Tensor,
                energies: torch.Tensor, forces: torch.Tensor,
                symm_func_params: dict[str, float],
                h: float) -> None:
                 
        super().__init__(cartesians, energies, forces, symm_func_params, h)
        self.g = g
        # enable gradient computation
        # because they are inputs of NN
        self.g.requires_grad = True
    
    def __getitem__(self, index):
        return self.cartesians[index], self.g[index], self.energies[index], self.forces[index]
    
def make_atomic_dataset(dataset: Subset,
                        symm_func_params: dict[str, float], h: float,
                        device: torch.device, train: bool = False) -> AtomicDataset:
    """Create atomic dataset with symmetric functions.

    Args:

        dataset (Dataset): dataset with atomic positions, energies and forces
        symm_func_params (dict[str, float]): parameters of symmetric functions
        h (float): step of coordinate-wise moving (used in forces caclulations).
        device (torch.device): device to store data
        train (bool): if True, create dataset for training

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
        g = calculate_g(cartesians, device, symm_func_params)
        return TrainAtomicDataset(cartesians, g, energies, forces, symm_func_params, h)
    # otherwise return just AtomicDataset
    else:
        return AtomicDataset(cartesians, energies, forces, symm_func_params, h)