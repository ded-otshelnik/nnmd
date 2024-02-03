import torch
from torch.utils.data import Dataset

class TrainAtomicDataset(Dataset):
    def __init__(self, cartesians: torch.Tensor, g: torch.Tensor,
                 e_dft: torch.Tensor, f_dft: torch.Tensor) -> None:
        self.cartesians = cartesians
        self.g = g
        self.e_dft = e_dft
        self.f_dft = f_dft
        self.len = self.cartesians.size(0)
    
    def __getitem__(self, index):
        return self.cartesians[index], self.g[index], self.e_dft[index], self.f_dft[index]
    
    def __len__(self):
        return self.len
    
class AtomicDataset(Dataset):
    def __init__(self, cartesians: torch.Tensor, g: torch.Tensor) -> None:
        self.cartesians = cartesians
        self.g = g
        self.len = self.cartesians.size(0)
    
    def __getitem__(self, index):
        return self.cartesians[index], self.g[index]
    
    def __len__(self):
        return self.len