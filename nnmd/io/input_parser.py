import yaml
import json

from .atomic_parser import traj_parser

def _parse_json_or_yaml(input_file: str) -> dict:
    with open(input_file, 'r') as f:
        if input_file.endswith('.json'):
            return json.load(f)
        elif input_file.endswith('.yaml') or input_file.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            raise ValueError('Unsupported file format: {}'.format(input_file))
    

def input_parser(input_file: str) -> dict:
    """Parses input file with atomic data and neural network parameters.
    It supports json and yaml formats.
    Args:
        input_file (str): Path to input file

    Raises:
        ValueError: Unsupported file format

    Returns:
        dict: Parsed data
    """
    input_data = _parse_json_or_yaml(input_file)
    n_atoms = None
    unit_cell = None
    for key, value in input_data.items():
        if key == "atomic_data" and isinstance(value, dict):
            for k, v in value.items():
                if k == "reference_data":
                    n_atoms, data, unit_cell = traj_parser(v)
                    input_data[key][k] = data
                elif k == "symmetry_functions_set":
                    symmetry_functions_data = _parse_json_or_yaml(v)
                    input_data[key][k] = {}
                    for element, functions in symmetry_functions_data['symmetry_functions_params'].items():
                        count = 0
                        features = []
                        params = []
                        h = None
                        for function, param_group in functions.items():
                            if function[0] == 'G':
                                for i in range(len(list(param_group.values())[0])):
                                    # for each set of parameters only number of function is needed
                                    features.append(int(function[1]))
                                    params.append([list(group) for group in zip(*param_group.values())][i])
                                    count += 1
                            elif function == 'h':
                                h = float(param_group)
                        input_data[key][k][element] = {"features": features, "params": params, "h": h, "n_features": count}

    input_data["atomic_data"]["n_atoms"] = n_atoms
    input_data["atomic_data"]["unit_cell"] = unit_cell

    return input_data