from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import Trajectory
from ase.md.verlet import VelocityVerlet
import ase.units as units

# Create a simple system of atoms
n_atoms = 3
traj_file = "input/lennard_jones.traj"
positions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
atoms = Atoms(f'Li{n_atoms}', positions=positions, pbc=True, cell=[10, 10, 10])

# Set the Lennard-Jones potential calculator
lj_calculator = LennardJones()

atoms.calc = lj_calculator

# Set up the MD simulation
timestep = 1  # time step of the simulation (in fs)
dyn = VelocityVerlet(atoms, timestep * units.fs)

# Saving the positions of all atoms after every time step
with Trajectory(traj_file, "w", atoms) as traj:
    dyn.attach(traj.write, interval=1)  # Save every step
    dyn.run(600)  # Run for 10 time steps
