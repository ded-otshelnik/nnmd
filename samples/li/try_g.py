import torch
from nnmd.util import input_parser, calculate_g, calculate_dg

input_data = input_parser('input/input.yaml')
cartesians = torch.tensor(input_data['atomic_data']['reference_data']['cartesians'], device = 'cuda', dtype = torch.float32)
g = calculate_g(cartesians, input_data['atomic_data']['symmetry_functions_set'][0]['Li'])
print(g)
dg = calculate_dg(cartesians, g, input_data['atomic_data']['symmetry_functions_set'][0]['Li'])
print(dg)