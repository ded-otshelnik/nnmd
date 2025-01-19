from nnmd.features._neighbors import CellNeighborList
from ase.io import Trajectory
import torch
import numpy as np

traj = Trajectory('input/Li_crystal_27.traj')

r_cutoff = 4.28

cartesians = torch.tensor(traj[1].positions, dtype = torch.float32)
cell = torch.tensor(traj[1].cell.array, dtype = torch.float32)
atomic_numbers = torch.tensor(traj[1].numbers, dtype = torch.float32)

cell_neighbor_list = CellNeighborList(r_cutoff)
mask = cell_neighbor_list(cartesians, atomic_numbers, cell)
print(mask)    