# MD simulation of Li atoms with neural network
import warnings
warnings.filterwarnings("ignore")

import torch

from nnmd.md import NNMD_calc
from nnmd.nn import BPNN
from nnmd.io import input_parser
from nnmd.features import calculate_params

from ase.md.verlet import VelocityVerlet
import ase.units as units

from ase.io import Trajectory

device = torch.device("cuda")
dtype = torch.float32

input_data = input_parser("input/input.yaml")

net = BPNN(dtype=dtype)
net.config(input_data["neural_network"])

sample_traj = Trajectory("input/Li_crystal_27.traj")
start = int(len(sample_traj) * 0.8)

atoms = sample_traj[start]

# ASE calculator
calc = NNMD_calc(
    model=net,
    properties=["energy", "forces"],
    symm_funcs_data=input_data["atomic_data"]["symmetry_functions_set"],
    atoms=atoms,
)

atoms.calc = calc

traj_file = f"Test_md_Li_27.traj"

# Moving to the MD part
timestep = 1  # time step of the simulation (in fs)

# Integrator for the equations of motion, timestep depends on system
dyn = VelocityVerlet(atoms, timestep * units.fs)

# Saving the positions of all atoms after every time step
with Trajectory(traj_file, "w", atoms) as traj:
    dyn.attach(traj.write, interval=1)
    # Running the simulation for timesteps
    dyn.run(600)
