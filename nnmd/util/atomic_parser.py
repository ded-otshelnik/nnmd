import re 
import traceback
from typing import Tuple

import numpy as np

from ase.io import Trajectory

def traj_parser(traj_file: str) -> tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
    """Extracts positions, energies, forces and velocities from ASE trajectory file

    """
    traj = Trajectory(traj_file)

    # for MD will be extracted only 4 types of characteristics 
    cartesians = []
    energies = []
    forces = []
    velocities = []

    # collect necessary data from each atomic structure
    for atoms in traj:
        cartesians.append(atoms.positions)
        energies.append(atoms.get_total_energy())
        forces.append(atoms.get_forces())
        velocities.append(atoms.get_velocities())

    return np.array(cartesians), np.array(energies), np.array(forces), np.array(velocities)

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