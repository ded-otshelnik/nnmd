import re 
import traceback
from typing import Tuple

def parser(filename, encoding='utf-8') -> Tuple[list, list, list, list]:
    with open(filename, encoding=encoding) as file:
        positions_marker, forces_marker = False, False
        cartesians, forces, energies, n_atoms = [], [], [], []

        line = file.readline() 
        while line:
            try:
                if line.startswith('Positions'):
                    positions_marker = True
                    line = file.readline()   
                    continue
                elif line.startswith('Forces in eV/Ang'):
                    forces_marker = True
                    line = file.readline()   
                    continue
                elif line.startswith('Extrapolated'):
                    energies.append(float(re.findall(r'[-+]?\d+.\d+', line)[0]))   
                    line = file.readline() 
                    continue
                elif line.startswith('Number of atoms:'):
                    n_atoms.append(int(re.findall(r'\d+', line)[0]))
                    line = file.readline()   
                    continue


                if positions_marker:
                    cartesians_iter = []
                    while line.strip('\n ') != '':
                        coord = re.findall(r'[^(,][-+]?\d+.\d+[^,)]', line)[:-3]
                        cartesians_iter.append([float(i) for i in coord])
                        line = file.readline() 
                    cartesians.append(cartesians_iter)
                    positions_marker = False
                elif forces_marker:
                    forces_iter = []
                    while line.strip('\n ') != '':
                        force = re.findall(r'[-+]?\d+.\d+', line)
                        forces_iter.append([float(i) for i in force])
                        line = file.readline() 
                    forces.append(forces_iter)    
                    forces_marker = False

                line = file.readline()    
            except Exception:
                traceback.print_exc()
                exit(1)
        
        return n_atoms, cartesians, forces, energies

if __name__ == '__main__':
    n_atoms, cartesians, forces, energies = parser('Cu111.txt')
    print(n_atoms)
    print(cartesians)
    print(forces)
    print(energies)