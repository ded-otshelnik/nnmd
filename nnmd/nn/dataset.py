import torch
from torch.utils.data import Dataset, Subset

from ..features import calculate_sf

class AtomicDataset(Dataset):
    def __init__(self, cartesians: torch.Tensor, 
                energies: torch.Tensor, forces: torch.Tensor,
                symm_func_data: dict) -> None:
        self.cartesians: torch.Tensor = cartesians
        self.energies: torch.Tensor = energies
        self.forces: torch.Tensor = forces

        self.symm_func_data: dict = symm_func_data
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

    def __getitem__(self, index):
        return self.g[index], self.dG[index], self.energies[index], self.forces[index]
    
def make_atomic_dataset(dataset: Subset,
                        symm_func_params: dict[str, float],
                        train: bool = True, **kwargs) -> AtomicDataset:
    """Create atomic dataset with symmetric functions.

    Args:

        dataset (Dataset): dataset with atomic positions, energies and forces
        symm_func_params (dict[str, float]): parameters of symmetric functions
        train (bool): if True, calculate g and dG

    Returns:
        AtomicDataset: atomic dataset with symmetric functions
    """
    # retrieve cartesians, energies, forces from subset and get symmetric functions values
    dataset_indices = dataset.indices
    cartesians = dataset.dataset.tensors[0][dataset_indices]
    energies = dataset.dataset.tensors[1][dataset_indices]
    forces = dataset.dataset.tensors[2][dataset_indices]

    # if dataset will be used
    # in training process
    # calculate g
    if train:
        path = kwargs['path']
        if kwargs['saved'] == True:
            g = torch.load(f'{path}_g.pt')
            dg = torch.load(f'{path}_dg.pt')
        else:
            g, dg = calculate_sf(cartesians, symm_func_params)
            torch.save(g, f'{path}_g.pt')
            torch.save(dg, f'{path}_dg.pt')
        return TrainAtomicDataset(cartesians, g, dg, energies, forces, symm_func_params)
    # otherwise return just AtomicDataset
    else:
        return AtomicDataset(cartesians, energies, forces, symm_func_params)