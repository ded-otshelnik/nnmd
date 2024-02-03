import re 
import traceback
from typing import Tuple

def gpaw_parser(filename, encoding='utf-8') -> Tuple[int, int, list, list, list]:
    """Parse info of gpaw simulation

    Args:
        filename: file of gpaw simulation
        encoding (str, optional): Defaults to 'utf-8'.
    """
    with open(filename, encoding=encoding) as file:
        # flags that marks positions and forces
        positions_marker, forces_marker = False, False
        cartesians, forces, energies, n_atoms = [], [], [], []

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
                # if atoms amount is found
                elif line.startswith('Number of atoms:'):
                    n_atoms = int(re.findall(r'\d+', line)[0])
                    line = file.readline()   
                    continue
                
                # parse atomic positions on iteration
                if positions_marker:
                    cartesians_iter = []
                    while line.strip('\n ') != '':
                        coord = re.findall(r'[^(,][-+]?\d+.\d+[^,)]', line)[:-3]
                        cartesians_iter.append([float(i) for i in coord])
                        line = file.readline() 
                    cartesians.append(cartesians_iter)
                    positions_marker = False

                # parse atomic forces on iteration
                elif forces_marker:
                    forces_iter = []
                    while line.strip('\n ') != '':
                        force = re.findall(r'[-+]?\d+.\d+', line)
                        forces_iter.append([float(i) for i in force])
                        line = file.readline() 
                    forces.append(forces_iter)    
                    forces_marker = False

                # move to next 
                line = file.readline()    
            except Exception:
                traceback.print_exc()
                exit(1)
        
        return len(cartesians), n_atoms, cartesians, forces, energies