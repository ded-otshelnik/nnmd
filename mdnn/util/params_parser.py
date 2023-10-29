from collections import namedtuple
import re 
import traceback

def parser(filename, encoding='utf-8')-> (list, list):
    # TODO: develop parser
    Point = namedtuple('Point', ['x', 'y', 'z'])

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
                break
        
        return cartesians, forces

if __name__ == '__main__':
    cartesians, forces = parser('test.txt')
    print(cartesians)
    print(forces)