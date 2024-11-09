import argparse
from calendar import c
import numpy as np
from ase.io import Trajectory

def get_info_from_traj(gpaw_file):
    traj = Trajectory(gpaw_file)
    cartesians = []
    forces = []
    velocities = []
    for atoms in traj:
        cartesians.append(atoms.positions)
        forces.append(atoms.get_forces())
        velocities.append(atoms.get_velocities())

    cartesians = np.array(cartesians)
    forces = np.array(forces)
    
    np.savez("cartesians_actual_coordwise.npy.npz", x = cartesians[:, :, 0],
                                                    y = cartesians[:, :, 1], 
                                                    z = cartesians[:, :, 2])
    np.savez("cartesians_actual.npy.npz", cartesians = cartesians)
    np.savez("forces_actual.npy.npz", forces = forces)
    np.savez("velocities_actual.npy.npz", velocities = velocities)

if __name__ == "__main__":
    params_parser = argparse.ArgumentParser(description = "Script extracts positions and forces for MD simulation with neural network")   
    params_parser.add_argument("traj_file", help = "File of ASE trajectory") 
    args = params_parser.parse_args()

    get_info_from_traj(args.traj_file)