from nnmd.features import calculate_sf, calculate_params
from nnmd.io import input_parser

import torch
import numpy as np

input_file = "input/input.yaml"
dataset = input_parser(input_file)
cartesians = []
for i, data in enumerate(dataset['atomic_data']['reference_data']):
    cartesians.append(data['Ag']['positions'])
cartesians = np.array(cartesians)

N1, N2, N3, N4, N5 = 0, 25, 0, 0, 25
r_cutoff = 12.0
params = calculate_params(r_cutoff, N1, N2, N3, N4, N5)
features = [1] * N1 + [2] * N2 + [3] * N3 + [4] * N4 + [5] * N5
dataset['atomic_data']['symmetry_functions_set']["Ag"] = {
    "params": params,
    "features": features
}

cartesians = torch.tensor(cartesians, dtype=torch.float32, device=torch.device("cuda"))
cell = torch.tensor(dataset['atomic_data']['unit_cell'], dtype=torch.float32, device=torch.device("cuda"))
symm_funcs_data = dataset['atomic_data']['symmetry_functions_set']

g, dg = calculate_sf(cartesians, cell, symm_funcs_data["Ag"])
print("g:", g)
print("dg:", dg)