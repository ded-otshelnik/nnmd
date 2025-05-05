# from ase.io import Trajectory
# import torch
# torch.manual_seed(0)
# import numpy as np
# from nnmd.features import calculate_sf, calculate_params

# traj = Trajectory('input/Li_crystal_27.traj')
# positions = np.array([atoms.positions for atoms in traj[1000:1001]])
# cell = torch.tensor(traj[1].cell.array, dtype = torch.float32, device = 'cuda')
# cartesians = torch.tensor(positions, dtype = torch.float32, device = 'cuda')
# cartesians.requires_grad = True

# r_cutoff = 38.0
# N_g1, N_g2, N_g3, N_g4, N_g5 = 0, 25, 0, 25, 0

# features = [1] * N_g1 + [2] * N_g2 + [3] * N_g3 + [4] * N_g4 + [5] * N_g5
# params = calculate_params(r_cutoff, N_g1, N_g2, N_g3, N_g4, N_g5)
# symm_funcs_data = {'features': features, 'params': params}

# g, dg = calculate_sf(cartesians, cell, symm_funcs_data)
# print('g.shape:', g.shape)

import json

def parse_symmetry_parameters(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    result = {}
    for element, symmetry_functions in data.items():
        result[element] = []
        for key, params in symmetry_functions.items():
            features = int(key[1])

            param_list = []
            max_length = max(len(value) if isinstance(value, list) else 1 for value in params.values())
            for i in range(max_length):
                param_set = {}
                for param_key, param_value in params.items():
                    if isinstance(param_value, list):
                        param_set[param_key] = param_value[i] if i < len(param_value) else param_value[-1]
                    else:
                        param_set[param_key] = param_value
                param_list.append(param_set)

            result[element].append({
                "features": features,
                "params": param_list
            })

    return result

from nnmd.io.input_parser import _symmetry_functions_parser, _parse_json_or_yaml

def parse_symmetry_functions(input_file):
    """Parse symmetry functions from a JSON or YAML file."""
    data = _parse_json_or_yaml(input_file)
    return _symmetry_functions_parser(data)

file_path = '/home/andrey-win/nnmd/samples/li/input/symm_li_27.json'
parsed_data = parse_symmetry_parameters(file_path)
print(json.dumps(parsed_data, indent=4))

print('------------------')
input_file = '/home/andrey-win/nnmd/samples/li/input/symm_li_27.json'
parsed_data = parse_symmetry_functions(input_file)
print(json.dumps(parsed_data, indent=4))
