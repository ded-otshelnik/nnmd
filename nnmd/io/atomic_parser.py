import re 
import traceback
from typing import Tuple

import numpy as np

from ase.io import Trajectory

def traj_parser(traj_file: str) -> tuple:
    """Extracts positions, energies, forces and velocities from ASE trajectory file
    with respect to each species

    Args:
        traj_file (str): ASE trajectory file

    Returns:
        list[int]: number of atoms in each species
        list[tuple[dict, np.ndarray]]: list of dictionaries with positions, forces and velocities
        for each species and energy of the system
    """
    traj = Trajectory(traj_file)

    # get species in dataset
    species = set(traj[0].symbols)
    # get unit cell
    cell = traj[0].get_cell().array

    # get number of atoms in each species
    n_atoms = list(len(traj[0][traj[0].symbols == spec]) for spec in species)

    # result list
    result = []

    # collect necessary data from each atomic structure and species
    # in a form of list of dictionaries
    for atoms in traj:        
        data = {}
        for spec in species:
            mask = atoms.symbols == spec
            data[spec] = {
                'positions': atoms.positions[mask]
            }
        data['forces'] = atoms.get_forces()
        data['velocities'] = atoms.get_velocities()
        data['energy'] = atoms.get_potential_energy()
        result.append(data)

    return n_atoms, result, cell

def gpaw_parser(filename, encoding = 'utf-8') -> Tuple[int, int, list, list, list]:
    """Parse info of gpaw simulation

    Args:
        filename: file of gpaw simulation
        encoding (str, optional): Defaults to 'utf-8'.
    """
    with open(filename, encoding = encoding) as file:
        # flags that marks positions and forces
        positions_marker, forces_marker = False, False
        cartesians, forces, energies = [], [], [],

        line = file.readline() 
        while line:
            try:
                # if cartesians values are found
                if line.startswith('Positions'):
                    # set a flag and move 
                    positions_marker = True
                    line = file.readline()   
                    continue
                # if forces values are found
                elif line.startswith('Forces in eV/Ang'):
                    # set a flag and move 
                    forces_marker = True
                    line = file.readline()   
                    continue
                # if energy value is found
                elif line.startswith('Extrapolated'):
                    energies.append(float(re.findall(r'[-+]?\d+.\d+', line)[0]))   
                    line = file.readline() 
                    continue
                
                # parse atomic positions on iteration
                if positions_marker:
                    cartesians_iter = []
                    while line.strip('\n ') != '':
                        coord = re.findall(r'[^(,][-+]?\d+.\d+[^,)]', line[4:])[:-3]
                        cartesians_iter.append([float(i) for i in coord])
                        line = file.readline() 
                    cartesians.append(cartesians_iter)
                    positions_marker = False

                # parse atomic forces on iteration
                elif forces_marker:
                    forces_iter = []
                    while line.strip('\n ') != '':
                        force = re.findall(r'[-+]?\d+.\d+', line[4:])
                        forces_iter.append([float(i) for i in force])
                        line = file.readline() 
                    forces.append(forces_iter)    
                    forces_marker = False

                # move to next 
                line = file.readline()    
            except Exception:
                traceback.print_exc()
                exit(1)
        
        return len(cartesians), len(cartesians[0]), cartesians, forces, energies