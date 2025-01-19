import torch
import numpy as np
from torch.utils.data import Dataset

from ..features import calculate_sf
    
class TrainAtomicDataset(Dataset):
    def __init__(self, cartesians: torch.Tensor, sf_data: dict[str, torch.Tensor],
                energies: torch.Tensor, forces: torch.Tensor,
                symm_func_params: dict[str, float]) -> None:
                 
        self.cartesians: torch.Tensor = cartesians
        self.energies: torch.Tensor = energies
        self.forces: torch.Tensor = forces

        self.sf_data: dict[str, torch.Tensor] = sf_data
        self.symm_func_data: dict = symm_func_params

        self.len = len(self.energies)

    def __getitem__(self, index):
        g = {spec: self.sf_data[spec][0][index] for spec in self.sf_data.keys()}
        dG = {spec: self.sf_data[spec][1][index] for spec in self.sf_data.keys()}
        return g, dG, self.energies[index], self.forces[index]
        
    def __len__(self):
        return self.len
    
def make_atomic_dataset(dataset: list, 
                        symm_func_params: dict[str, dict[str, list[float | int]]],
                        device: torch.device | str, **kwargs) -> TrainAtomicDataset:
    """Create atomic dataset with symmetric functions.

    Args:
        dataset (list): list with positions by species, unit cell, forces and velocities.
        symm_func_params (dict[str, dict[str, list[float | int]]]): parameters of symmetric functions.
        device (torch.device | str, optional): device for torch tensors. Defaults to torch.device('cuda').

    Returns:
        TrainAtomicDataset: dataset with symmetric functions.
    """
    cell = torch.as_tensor(dataset['unit_cell'], dtype = torch.float32, device = device)
    dataset = dataset['reference_data']
    # convert data to torch tensors
    energies = torch.tensor(np.array([data['energy'] for data in dataset]), dtype = torch.float32, device = device)
    forces = torch.tensor(np.array([data['forces'] for data in dataset]), dtype = torch.float32, device = device)
    
    # get cartesian coordinates for each species
    cartesians = {spec: torch.tensor(np.array([data[spec]['positions'] for data in dataset]), dtype = torch.float32, device = device)
                   for spec in dataset[0].keys() if spec not in ['forces', 'energy', 'velocities']}
    
    
    sf_data = {}
    for spec in cartesians.keys():
        if 'saved' in kwargs and kwargs['saved'] == True:
            g_spec = torch.load(f'g_{spec}.pt', map_location = device)
            dg_spec = torch.load(f'dg_{spec}.pt', map_location = device)
        else:
            g_spec, dg_spec = calculate_sf(cartesians[spec], cell, symm_func_params[spec])
            if 'saved' in kwargs and kwargs['saved'] == False:
                torch.save(g_spec, f'g_{spec}.pt')
                torch.save(dg_spec, f'dg_{spec}.pt')
        if g_spec.requires_grad != True: 
            g_spec.requires_grad = True
        g_spec = (g_spec - g_spec.mean()) / g_spec.std()
        assert torch.isnan(g_spec).sum() == 0, f"Symmetry functions for {spec} contain NaN values"
        sf_data[spec] = (g_spec, dg_spec)

    return TrainAtomicDataset(cartesians, sf_data, energies, forces, symm_func_params)