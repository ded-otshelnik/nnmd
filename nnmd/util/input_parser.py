import yaml
import json

from nnmd.util import traj_parser

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

    for key, value in input_data.items():
        print(key, value)
        if key == "atomic_data" and isinstance(value, dict):
            for k, v in value.items():
                if k == "reference_data":
                    cartesians, energies, forces, velocities = traj_parser(v)
                    input_data[key][k] = {"cartesians": cartesians,
                                            "energies": energies,
                                            "forces": forces,
                                            "velocities": velocities}
                elif k == "symmetry_functions_set":
                    symmetry_functions_data = _parse_json_or_yaml(v)
                    input_data[key][k] = symmetry_functions_data

    return input_data