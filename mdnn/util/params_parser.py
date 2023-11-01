import re 
import traceback

def parser(filename, encoding='utf-8')-> (list, list, list):
    n_atoms_iter = []
    with open(filename, encoding=encoding) as file:
        positions_marker, energies, forces_marker = False, False, False
        cartesians, forces = [], []

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


                if positions_marker:
                    cartesians_iter = []
                    n_atoms = 0
                    while line.strip('\n ') != '':
                        coord = re.findall(r'[^(,][-+]?\d+.\d+[^,)]', line)[:-3]
                        cartesians_iter.append([float(i) for i in coord])
                        line = file.readline() 
                        n_atoms += 1
                    cartesians.append(cartesians_iter)
                    n_atoms_iter.append(n_atoms)
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
        
        return n_atoms, cartesians, forces

if __name__ == '__main__':
    n_atoms, cartesians, forces = parser('test.txt')
    print(n_atoms)
    print(cartesians)
    print(forces)