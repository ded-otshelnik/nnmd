import yaml
import json

from .gpaw import gpaw_parser
from .ase import traj_parser


def input_parser(input_file: str) -> dict:
    """Parses input file with atomic data and neural network parameters.
    It supports json and yaml formats.

    Args:
        input_file (str): Path to input file

    Raises:
        ValueError: Unsupported file format

    Returns:
        dict: Parsed atomic data and neural network parameters.
    """
    input_data = _parse_json_or_yaml(input_file)

    n_atoms = None
    unit_cell = None
    pbc = None

    for key, value in input_data.items():
        if key == "atomic_data" and isinstance(value, dict):
            for k, v in value.items():
                if k == "reference_data":
                    if v.endswith(".traj"):
                        n_atoms, data, unit_cell, pbc = traj_parser(v)
                        input_data[key][k] = data
                    elif v.endswith("txt"):
                        n_atoms, data, unit_cell, pbc = gpaw_parser(v)
                        input_data[key][k] = data
                    else:
                        raise ValueError(
                            "Unsupported reference data format: {}".format(v)
                        )
                elif k == "symmetry_functions_set":
                    symmetry_functions_data = _parse_json_or_yaml(v)
                    input_data[key][k] = _symmetry_functions_parser(
                        symmetry_functions_data
                    )

                    # input sizes for each element depend on symmetry functions count
                    input_data["neural_network"]["input_sizes"] = _get_input_sizes(
                        input_data[key][k]
                    )

    input_data["atomic_data"]["n_atoms"] = n_atoms
    input_data["atomic_data"]["unit_cell"] = unit_cell
    input_data["atomic_data"]["pbc"] = pbc

    return input_data


def _parse_json_or_yaml(input_file: str) -> dict:
    with open(input_file, "r") as f:
        if input_file.endswith(".json"):
            return json.load(f)
        elif input_file.endswith(".yaml") or input_file.endswith(".yml"):
            return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format: {}".format(input_file))


def _symmetry_functions_parser(symmetry_functions_data: dict) -> dict:
    """Parses symmetry functions data from the input file.

    Args:
        symmetry_functions_data (dict): Symmetry functions data

    Returns:
        dict: Parsed symmetry functions data
    """
    parsed_data = {}
    for element, functions in symmetry_functions_data.items():
        features_list = []
        param_list = []
        for key, params in functions.items():

            # Get lengths of the parameters for zipping
            max_length = max(
                len(value) if isinstance(value, list) else 1
                for value in params.values()
            )

            for i in range(max_length):
                params_group = []
                for param_value in params.values():
                    if isinstance(param_value, list):
                        params_group.append(
                            param_value[i] if i < len(param_value) else param_value[-1]
                        )
                    else:
                        params_group.append(param_value)

                features_list.append(int(key[1]))
                param_list.append(params_group)

        parsed_data[element] = {
            "features": features_list,
            "params": param_list,
            "n_features": len(param_list),
        }
    return parsed_data


def _get_input_sizes(features: dict) -> list:
    """Calculates input sizes for each element in the symmetry functions data.

    Args:
        features (dict): Symmetry functions data

    Returns:
        list: List of input sizes for each element
    """
    input_sizes = []
    for _, data in features.items():
        n_features = data["n_features"]
        input_sizes.append(n_features)
    return input_sizes
