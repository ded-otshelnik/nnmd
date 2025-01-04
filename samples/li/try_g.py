import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from nnmd.util import input_parser, calculate_g, calculate_dg

input_data = input_parser('input/input.yaml')
cartesians = torch.tensor(input_data['atomic_data']['reference_data']['cartesians'], device = 'cpu', dtype = torch.float32)[:10]
g = calculate_g(cartesians, input_data['atomic_data']['symmetry_functions_set'][0]['Li'])
print(g)