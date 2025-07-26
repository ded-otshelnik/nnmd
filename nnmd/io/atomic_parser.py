import re
import traceback

import numpy as np
from ase.io import Trajectory


def traj_parser(traj_file: str) -> tuple:
    """Extracts positions, energies, forces and velocities from ASE trajectory file
    with respect to each species.

    Args:
        traj_file (str): ASE trajectory file.

    Returns:
        tuple: number of atoms in each species, list of dictionaries with
        positions, forces, velocities and energies for each atomic structure,
        unit cell.
    """
    traj = Trajectory(traj_file)

    # get species in dataset
    species = set(traj[0].symbols)
    # get unit cell
    cell = traj[0].get_cell().array

    # get number of atoms in each species
    n_atoms = list(len(traj[0][traj[0].symbols == spec]) for spec in species)

    result = []

    # collect necessary data from each atomic structure and species
    # in a form of list of dictionaries
    for atoms in traj[1:]:
        data = {}
        for spec in species:
            mask = atoms.symbols == spec
            data[spec] = {"positions": atoms.positions[mask]}
        data["forces"] = atoms.get_forces()
        data["velocities"] = atoms.get_velocities()
        data["energy"] = atoms.get_potential_energy()
        result.append(data)

    return n_atoms, result, cell


def gpaw_parser(filename, encoding="utf-8") -> tuple:
    """Parse info of gpaw simulation.

    Args:
        filename: file of gpaw simulation.
        encoding (str, optional): Defaults to 'utf-8'.

    Returns:
        tuple: number of atoms, list of dictionaries with positions, forces and energy,
        unit cell.
    """

    with open(filename, encoding=encoding) as file:
        # flags that marks positions and forces
        positions_marker, forces_marker = False, False

        result = []
        cell = None

        line = file.readline()
        data = {}
        while line:
            try:
                # if cartesians values are found
                if line.startswith("Positions"):
                    # set a flag and move
                    positions_marker = True
                    line = file.readline()
                    continue
                # if forces values are found
                elif line.startswith("Forces in eV/Ang"):
                    # set a flag and move
                    forces_marker = True
                    line = file.readline()
                    continue

                # if energy value is found
                elif line.startswith("Extrapolated"):
                    energy = float(re.findall(r"[-+]?\d+.\d+", line)[0])
                    data["energy"] = energy
                    line = file.readline()
                    continue

                if cell is None and line.startswith("Unit cell"):
                    cell = []
                    while len(line.strip("\n ")) != 0:
                        line = file.readline()
                        matched_group = re.findall(r"[-+]?\d+.\d+", line)[:3]
                        if matched_group:
                            cell.append(list(map(float, matched_group)))
                    cell = np.array(cell).reshape(3, 3)
                    line = file.readline()
                    continue

                # parse atomic positions on iteration
                if positions_marker:
                    while line.strip("\n ") != "":
                        coord = re.findall(r"[^(,][-+]?\d+.\d+[^,)]", line[4:])[:-3]
                        spec = re.findall(r"\w+", line[4:])[0]
                        if spec not in data:
                            data[spec] = {"positions": []}
                        data[spec]["positions"].append([float(i) for i in coord])
                        line = file.readline()

                    data[spec]["positions"] = np.array(data[spec]["positions"])
                    positions_marker = False

                # parse atomic forces on iteration
                elif forces_marker:
                    forces_iter = []
                    while line.strip("\n ") != "":
                        force = re.findall(r"[-+]?\d+.\d+", line[4:])
                        forces_iter.append([float(i) for i in force])
                        line = file.readline()

                    data["forces"] = forces_iter
                    forces_marker = False

                # if data is collected (positions, forces and energy)
                if len(data.keys()) == 3:
                    result.append(data)
                    data = {}

                line = file.readline()
            except Exception:
                traceback.print_exc()
                exit(1)

        # get number of atoms
        n_atoms = len(result[0]["forces"])
        return n_atoms, result, cell
